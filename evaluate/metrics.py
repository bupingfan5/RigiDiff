import os
import glob
import re
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, AllChem

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RDLogger.DisableLog("rdApp.*")


try:
    from src import sascorer
except ImportError:
    sascorer = None


def calculate_sa_score(mol):
    'Compute the normalized synthetic-accessibility score.'
    if sascorer is None:
        return np.nan

    try:
        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if canonical_mol is None:
            canonical_mol = mol

        raw_sa = float(sascorer.calculateScore(canonical_mol))
        sa_score = (10.0 - raw_sa) / 9.0
        return round(float(np.clip(sa_score, 0.0, 1.0)), 2)
    except Exception:
        return np.nan


def count_lipinski_rules(mol):
    'Count satisfied Lipinski-style drug-likeness rules.'
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        score = 0
        if mw <= 500: score += 1
        if logp <= 5: score += 1
        if hbd <= 5: score += 1
        if hba <= 10: score += 1
        if rot_bonds <= 10: score += 1

        return score
    except:
        return 0


def calculate_diversity(fps):
    """
    Calculate mean pairwise Tanimoto dissimilarity within one pocket.

    This matches the TargetDiff, PMDM, and GCDM evaluation protocol:
    default RDKit topological fingerprints and all unordered molecule pairs.
    """
    if len(fps) < 2:
        return 0.0

    dissimilarities = []
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        dissimilarities.extend(1.0 - sim for sim in sims)

    return float(np.mean(dissimilarities))


def extract_pocket_id(file_name):
    """Extract the pocket identifier from target_<id>_sample_<id>.sdf."""
    match = re.match(r"^target_(.+?)_sample_", file_name)
    if match is None:
        raise ValueError(
            "Cannot infer pocket ID from SDF filename "
            f"'{file_name}'. Expected target_<id>_sample_<id>.sdf."
        )
    return match.group(1)


def evaluate_folder(sdf_dir, output_csv="metrics_standard_sa.csv"):
    
    if not os.path.exists(sdf_dir):
        print(f"[Error] Directory does not exist: {sdf_dir}")
        return

    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    print(f"[Start] Found {len(sdf_files)} molecule files; running standard evaluation...")

    results = []
    fps_by_pocket = defaultdict(list)
    result_indices_by_pocket = defaultdict(list)

    # ==========================================
    
    # ==========================================
    for f in tqdm(sdf_files, desc="1/2 Evaluating molecular properties", file=sys.stdout):
        file_name = os.path.basename(f)
        pocket_id = extract_pocket_id(file_name)

        suppl = Chem.SDMolSupplier(f, sanitize=False)

        for mol in suppl:
            if mol is None: continue

            try:

                Chem.SanitizeMol(mol)

                
                sa_val = calculate_sa_score(mol)

                
                qed_val = QED.qed(mol)
                lipinski_score = count_lipinski_rules(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)

                
                num_rings = mol.GetRingInfo().NumRings()

                
                num_bonds = mol.GetNumBonds()
                bond_single = 0
                bond_double = 0
                bond_triple = 0
                bond_aromatic = 0

                for bond in mol.GetBonds():
                    bt = bond.GetBondType()
                    if bt == Chem.rdchem.BondType.SINGLE:
                        bond_single += 1
                    elif bt == Chem.rdchem.BondType.DOUBLE:
                        bond_double += 1
                    elif bt == Chem.rdchem.BondType.TRIPLE:
                        bond_triple += 1
                    elif bt == Chem.rdchem.BondType.AROMATIC:
                        bond_aromatic += 1

                
                fp = Chem.RDKFingerprint(mol)
                result_index = len(results)
                fps_by_pocket[pocket_id].append(fp)
                result_indices_by_pocket[pocket_id].append(result_index)

                results.append({
                    "file": file_name,
                    "QED": qed_val,
                    "SA_Score": sa_val,
                    "Lipinski": lipinski_score,
                    "MW": mw,
                    "LogP": logp,
                    "Atoms": mol.GetNumAtoms(),
                    "Rings": num_rings,  
                    "Total_Bonds": num_bonds,  
                    "Single_Bonds": bond_single,  
                    "Double_Bonds": bond_double,  
                    "Triple_Bonds": bond_triple,  
                    "Aromatic_Bonds": bond_aromatic  
                })
            except Exception as e:
                continue

    if not results:
        print("[Warning] No molecules were evaluated. Check that the SDF files are valid.")
        return

    # ==========================================
    
    # ==========================================
    print("\n[Progress] 2/2 Computing per-pocket structural diversity...")
    per_pocket_diversity = {}
    for pocket_id, pocket_fps in tqdm(
        fps_by_pocket.items(),
        desc="Calculating Diversity",
        file=sys.stdout,
    ):
        diversity = calculate_diversity(pocket_fps)
        per_pocket_diversity[pocket_id] = diversity
        for result_index in result_indices_by_pocket[pocket_id]:
            results[result_index]["Diversity"] = diversity

    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # ==========================================
    
    # ==========================================
    diversity_values = np.asarray(list(per_pocket_diversity.values()), dtype=float)
    macro_diversity = float(np.mean(diversity_values))
    median_diversity = float(np.median(diversity_values))
    std_diversity = float(np.std(diversity_values))

    print("\n" + "=" * 55)
    print("Standard Metrics")
    print("=" * 55)
    print(f"Successful samples: {len(results)}")
    print(f"Mean QED: {df['QED'].mean():.4f}")
    print(f"Mean SA (0-1): {df['SA_Score'].mean():.4f} (higher is better)")
    print(f"Mean Lipinski score: {df['Lipinski'].mean():.2f} / 5.0")
    print(f"Mean molecular weight: {df['MW'].mean():.2f}")
    print(f"Mean ring count: {df['Rings'].mean():.2f}")
    print(f"Mean bond count: {df['Total_Bonds'].mean():.2f}")
    print(f"Per-pocket diversity (macro mean): {macro_diversity:.4f} (higher is better)")
    print(f"Per-pocket diversity (median): {median_diversity:.4f}")
    print(f"Per-pocket diversity (std): {std_diversity:.4f}")
    print(f"Pockets included in diversity aggregation: {len(per_pocket_diversity)}")
    print("=" * 55)
    print(f"Saved full results to: {output_csv}")


if __name__ == "__main__":
    
    SDF_PATH = "samples/final run/candidates"
    evaluate_folder(SDF_PATH)
