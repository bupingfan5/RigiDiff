from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, RDConfig

from case_utils import CASE_ROOT, load_config, read_first_sdf


PROTEIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}
PROTEIN_ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "PHE", "TRP", "TYR"}
POSITIVE_RESIDUES = {"LYS": {"NZ"}, "ARG": {"CZ", "NH1", "NH2"}}
NEGATIVE_RESIDUES = {"ASP": {"CG", "OD1", "OD2"}, "GLU": {"CD", "OE1", "OE2"}}
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def protein_records(path: Path):
    structure = PDBParser(QUIET=True).get_structure("protein", str(path))
    atoms = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].strip():
                    continue
                record = {
                    "chain": chain.id,
                    "resname": residue.resname.strip(),
                    "resnum": int(residue.id[1]),
                    "atoms": {},
                }
                for atom in residue:
                    element = (atom.element or "").strip().title()
                    if element == "H":
                        continue
                    item = {
                        "chain": chain.id,
                        "resname": residue.resname.strip(),
                        "resnum": int(residue.id[1]),
                        "atom_name": atom.name.strip(),
                        "element": element,
                        "coord": np.asarray(atom.coord, dtype=float),
                    }
                    atoms.append(item)
                    record["atoms"][atom.name.strip()] = item
                residues.append(record)
        break
    return atoms, residues


def ligand_features(mol: Chem.Mol):
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    features = factory.GetFeaturesForMol(mol)
    donors, acceptors, positives, negatives = set(), set(), set(), set()
    for feature in features:
        atom_ids = set(feature.GetAtomIds())
        family = feature.GetFamily()
        if family == "Donor":
            donors.update(atom_ids)
        elif family == "Acceptor":
            acceptors.update(atom_ids)
        elif family == "PosIonizable":
            positives.update(atom_ids)
        elif family == "NegIonizable":
            negatives.update(atom_ids)
    return donors, acceptors, positives, negatives


def ligand_coordinates(mol: Chem.Mol):
    conf = mol.GetConformer()
    return np.asarray(
        [
            [
                conf.GetAtomPosition(index).x,
                conf.GetAtomPosition(index).y,
                conf.GetAtomPosition(index).z,
            ]
            for index in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def ring_geometry(coords: np.ndarray):
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= max(np.linalg.norm(normal), 1e-8)
    return center, normal


def append_contact(rows, ligand, atom, protein_atom, interaction_type, distance):
    rows.append(
        {
            "ligand": ligand,
            "residue_name": protein_atom["resname"],
            "residue_number": protein_atom["resnum"],
            "chain": protein_atom["chain"],
            "interaction_type": interaction_type,
            "distance": float(distance),
            "ligand_atom": f"{atom.GetSymbol()}{atom.GetIdx() + 1}",
            "protein_atom": protein_atom["atom_name"],
            "detection_source": "internal_geometry_v1",
        }
    )


def analyze_ligand(name: str, ligand_path: Path, protein_atoms, protein_residues):
    mol = read_first_sdf(ligand_path, remove_hs=False)
    coords = ligand_coordinates(mol)
    donors, acceptors, positives, negatives = ligand_features(mol)
    rows = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        index = atom.GetIdx()
        coord = coords[index]
        for protein_atom in protein_atoms:
            distance = np.linalg.norm(coord - protein_atom["coord"])
            resname = protein_atom["resname"]
            atom_name = protein_atom["atom_name"]
            protein_is_donor = atom_name == "N" or atom_name in PROTEIN_DONORS.get(resname, set())
            protein_is_acceptor = atom_name == "O" or atom_name in PROTEIN_ACCEPTORS.get(resname, set())

            if distance <= 3.5 and (
                (index in donors and protein_is_acceptor)
                or (index in acceptors and protein_is_donor)
            ):
                append_contact(
                    rows, name, atom, protein_atom, "hydrogen_bond_candidate", distance
                )
            if (
                distance <= 4.0
                and atom.GetAtomicNum() in {6, 9, 17, 35, 53}
                and resname in HYDROPHOBIC_RESIDUES
                and protein_atom["element"] in {"C", "S"}
                and atom_name not in {"C", "CA"}
            ):
                append_contact(rows, name, atom, protein_atom, "hydrophobic", distance)
            if (
                distance <= 4.0
                and index in positives
                and atom_name in NEGATIVE_RESIDUES.get(resname, set())
            ) or (
                distance <= 4.0
                and index in negatives
                and atom_name in POSITIVE_RESIDUES.get(resname, set())
            ):
                append_contact(rows, name, atom, protein_atom, "salt_bridge_candidate", distance)
            if (
                atom.GetAtomicNum() in {9, 17, 35, 53}
                and protein_atom["element"] in {"N", "O", "S"}
                and distance <= 3.5
            ):
                append_contact(rows, name, atom, protein_atom, "halogen_bond_candidate", distance)

    ligand_rings = []
    for ring in Chem.GetSymmSSSR(mol):
        indices = list(ring)
        if all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in indices):
            center, normal = ring_geometry(coords[indices])
            ligand_rings.append((indices, center, normal))

    protein_rings = []
    for residue in protein_residues:
        names = AROMATIC_RING_ATOMS.get(residue["resname"])
        if not names or not all(name in residue["atoms"] for name in names):
            continue
        ring_coords = np.asarray([residue["atoms"][name]["coord"] for name in names])
        center, normal = ring_geometry(ring_coords)
        protein_rings.append((residue, center, normal))

    for indices, lig_center, lig_normal in ligand_rings:
        ligand_atom = mol.GetAtomWithIdx(indices[0])
        for residue, prot_center, prot_normal in protein_rings:
            distance = np.linalg.norm(lig_center - prot_center)
            angle = np.degrees(
                np.arccos(np.clip(abs(np.dot(lig_normal, prot_normal)), 0.0, 1.0))
            )
            if distance <= 5.5 and (angle <= 30.0 or angle >= 60.0):
                protein_atom = residue["atoms"][next(iter(residue["atoms"]))]
                append_contact(
                    rows,
                    name,
                    ligand_atom,
                    protein_atom,
                    "pi_stacking_candidate",
                    distance,
                )
        for residue in protein_residues:
            for atom_names in POSITIVE_RESIDUES.values():
                for atom_name in atom_names:
                    protein_atom = residue["atoms"].get(atom_name)
                    if protein_atom is None or residue["resname"] not in POSITIVE_RESIDUES:
                        continue
                    distance = np.linalg.norm(lig_center - protein_atom["coord"])
                    if distance <= 6.0:
                        append_contact(
                            rows,
                            name,
                            ligand_atom,
                            protein_atom,
                            "pi_cation_candidate",
                            distance,
                        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        subset=[
            "ligand",
            "residue_name",
            "residue_number",
            "chain",
            "interaction_type",
            "ligand_atom",
            "protein_atom",
        ]
    ).sort_values(["interaction_type", "distance"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    args = parser.parse_args()
    config = load_config()
    directory = target_dir(config, args.target_id)
    selected = pd.read_csv(directory / "properties" / "case_molecules.csv")
    protein_atoms, protein_residues = protein_records(
        directory / "reference" / "pocket.pdb"
    )

    inputs = {
        "Reference 524": directory / "reference" / "crossdocked_reference.sdf",
        "Candidate A": Path(
            selected.loc[selected["role"].eq("Candidate A"), "selected_docked_sdf"].iloc[0]
        ),
        "Candidate B": Path(
            selected.loc[selected["role"].eq("Candidate B"), "selected_docked_sdf"].iloc[0]
        ),
    }
    frames = [
        analyze_ligand(name, path, protein_atoms, protein_residues)
        for name, path in inputs.items()
    ]
    interactions = pd.concat(frames, ignore_index=True)
    output_dir = directory / "interactions"
    output_dir.mkdir(exist_ok=True)
    interactions.to_csv(output_dir / "case_interactions.csv", index=False)

    if interactions.empty:
        summary = pd.DataFrame(
            columns=["ligand", "interaction_type", "interaction_count", "residues"]
        )
    else:
        summary = (
            interactions.groupby(["ligand", "interaction_type"])
            .agg(
                interaction_count=("interaction_type", "size"),
                residues=(
                    "residue_number",
                    lambda values: ";".join(
                        map(str, sorted(set(int(value) for value in values)))
                    ),
                ),
            )
            .reset_index()
        )
    summary.to_csv(output_dir / "interaction_summary.csv", index=False)

    metadata = {
        "backend": "internal_geometry_v1",
        "receptor_scope": "the exact cropped pocket used by the generation model",
        "interpretation": (
            "Hydrogen-bond, salt-bridge, halogen-bond, pi-stacking, and "
            "pi-cation rows are geometry-based candidates and require visual "
            "inspection or confirmation with PLIP/ProLIF."
        ),
        "hydrogen_bond_cutoff_angstrom": 3.5,
        "hydrophobic_cutoff_angstrom": 4.0,
        "pi_stacking_cutoff_angstrom": 5.5,
        "pi_cation_cutoff_angstrom": 6.0,
    }
    (output_dir / "interaction_method.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(interactions)} interaction records")


if __name__ == "__main__":
    main()
