from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors


CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_ROOT.parent
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RDLogger.DisableLog("rdApp.warning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src import sascorer
except Exception:
    sascorer = None


def load_config() -> dict:
    path = CASE_ROOT / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    root = Path(config.get("project_root", PROJECT_ROOT)).resolve()
    config["project_root"] = root
    return config


def project_path(config: dict, key: str) -> Path:
    value = Path(config[key])
    return value if value.is_absolute() else config["project_root"] / value


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sample_stem(row: pd.Series | dict) -> str:
    for key in ("sdf_path", "file", "source_file"):
        value = row.get(key, "")
        if isinstance(value, str) and value:
            match = re.search(r"(target_\d+_sample_\d+)", Path(value).name)
            if match:
                return match.group(1)
    source_idx = row.get("source_sample_idx")
    target_id = row.get("target_id", row.get("dataset_target_index"))
    if pd.notna(source_idx):
        return f"target_{int(target_id)}_sample_{int(source_idx)}"
    raise ValueError(f"Cannot determine sample stem from row: {row}")


def parse_chain_from_pocket(path: str) -> str:
    name = Path(path).name
    match = re.match(r"[0-9A-Za-z]{4}_([^_]+)_rec_", name)
    return match.group(1) if match else ""


def crossdocked_reference_path(config: dict, pocket_rel: str) -> Path:
    rel = re.sub(r"_pocket10\.pdb$", ".sdf", pocket_rel, flags=re.IGNORECASE)
    return project_path(config, "data_root") / rel


def generated_sdf_path(config: dict, stem: str) -> Path:
    work = project_path(config, "vina_work_dir")
    explicit_h = work / "ligands_pdbqt" / f"{stem}_explicitH.sdf"
    if explicit_h.exists():
        return explicit_h
    raise FileNotFoundError(f"Generated SDF not found for {stem}: {explicit_h}")


def docked_pdbqt_path(config: dict, stem: str) -> Path:
    path = project_path(config, "vina_work_dir") / "vina_outputs" / f"{stem}_full_dock.pdbqt"
    if not path.exists():
        raise FileNotFoundError(f"Docked PDBQT not found for {stem}: {path}")
    return path


def read_first_sdf(path: Path, remove_hs: bool = False) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=remove_hs)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule: {path}")
    Chem.SanitizeMol(mol)
    return mol


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def convert_pdbqt_to_sdf(config: dict, source: Path, destination: Path) -> Path:
    """Restore the first Vina pose using Meeko's SMILES-to-PDBQT atom mapping.

    This preserves the generated molecular graph and bond orders. Open Babel
    reconstruction from PDBQT is intentionally avoided because PDBQT does not
    encode bond orders directly.
    """
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    smiles = None
    mapping_tokens: list[int] = []
    coordinates: dict[int, tuple[float, float, float]] = {}
    in_first_model = False
    for line in lines:
        if line.startswith("MODEL"):
            if in_first_model:
                break
            in_first_model = True
            continue
        if line.startswith("ENDMDL") and in_first_model:
            break
        if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
            smiles = line[len("REMARK SMILES ") :].strip()
        elif line.startswith("REMARK SMILES IDX"):
            mapping_tokens.extend(
                int(token) for token in line[len("REMARK SMILES IDX") :].split()
            )
        elif line.startswith(("ATOM  ", "HETATM")):
            serial = int(line[6:11])
            coordinates[serial] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    if not smiles:
        raise ValueError(f"Missing REMARK SMILES in {source}")
    if len(mapping_tokens) % 2:
        raise ValueError(f"Malformed REMARK SMILES IDX mapping in {source}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse PDBQT SMILES in {source}: {smiles}")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    mapped_atoms = set()
    for smiles_index, pdbqt_serial in zip(mapping_tokens[0::2], mapping_tokens[1::2]):
        atom_index = smiles_index - 1
        if atom_index < 0 or atom_index >= mol.GetNumAtoms():
            raise ValueError(f"SMILES atom index out of range in {source}: {smiles_index}")
        if pdbqt_serial not in coordinates:
            raise ValueError(f"PDBQT atom serial missing in {source}: {pdbqt_serial}")
        x, y, z = coordinates[pdbqt_serial]
        conformer.SetAtomPosition(atom_index, (x, y, z))
        mapped_atoms.add(atom_index)
    if len(mapped_atoms) != mol.GetNumAtoms():
        missing = sorted(set(range(mol.GetNumAtoms())) - mapped_atoms)
        raise ValueError(f"Incomplete PDBQT atom mapping in {source}; missing {missing}")
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", source.stem)
    mol.SetProp("pose_source", "AutoDock Vina first-ranked pose")
    write_sdf(mol, destination)
    return destination


def sa_paper_from_raw(raw_sa: float) -> float:
    return float(np.clip((10.0 - float(raw_sa)) / 9.0, 0.0, 1.0))


def lipinski_score(mol: Chem.Mol) -> int:
    mol = Chem.RemoveHs(mol)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return int(mw <= 500) + int(logp <= 5) + int(hbd <= 5) + int(hba <= 10) + int(rotatable <= 10)


def molecular_properties(
    mol: Chem.Mol,
    *,
    existing_qed: float | None = None,
    existing_raw_sa: float | None = None,
    existing_logp: float | None = None,
    existing_mw: float | None = None,
) -> dict:
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    if existing_raw_sa is None and sascorer is not None:
        canonical = Chem.MolFromSmiles(smiles)
        existing_raw_sa = float(sascorer.calculateScore(canonical))
    return {
        "canonical_smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": float(existing_mw) if existing_mw is not None and math.isfinite(existing_mw) else float(Descriptors.MolWt(mol)),
        "qed": float(existing_qed) if existing_qed is not None and math.isfinite(existing_qed) else float(QED.qed(mol)),
        "sa_raw": float(existing_raw_sa) if existing_raw_sa is not None and math.isfinite(existing_raw_sa) else np.nan,
        "sa": sa_paper_from_raw(existing_raw_sa) if existing_raw_sa is not None and math.isfinite(existing_raw_sa) else np.nan,
        "logp": float(existing_logp) if existing_logp is not None and math.isfinite(existing_logp) else float(Crippen.MolLogP(mol)),
        "lipinski": lipinski_score(mol),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(Lipinski.NumAromaticRings(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
    }


def fingerprint(mol: Chem.Mol):
    return MORGAN_GENERATOR.GetFingerprint(Chem.RemoveHs(mol))


def tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return float(DataStructs.TanimotoSimilarity(fingerprint(mol_a), fingerprint(mol_b)))


def diversity_from_smiles(smiles_values: Iterable[str]) -> float:
    mols = []
    seen = set()
    for value in smiles_values:
        if not isinstance(value, str) or not value:
            continue
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        mols.append(mol)
    if len(mols) < 2:
        return float("nan")
    fps = [fingerprint(mol) for mol in mols]
    similarities = []
    for index, fp in enumerate(fps[:-1]):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fp, fps[index + 1 :]))
    return float(1.0 - np.mean(similarities))


def rank_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=higher_is_better, method="average")


def load_vina(config: dict) -> pd.DataFrame:
    frame = pd.read_csv(project_path(config, "vina_csv"), low_memory=False)
    frame["target_id"] = numeric(frame["target_id"]).astype("Int64")
    frame["vina"] = numeric(frame["full_protein_vina_score"])
    for column in ("qed", "sa", "logp", "mw", "vdw_overlap_max", "minimum_relative_distance"):
        if column in frame:
            frame[column] = numeric(frame[column])
    return frame


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_text(value, default="--") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return default
    text = str(value)
    return default if text.lower() == "nan" or text == "" else text
