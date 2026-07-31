from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem

from case_utils import CASE_ROOT, load_config, read_first_sdf


VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}


def structure_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    coords, atomic_numbers = [], []
    periodic = Chem.GetPeriodicTable()
    for atom in structure.get_atoms():
        element = (atom.element or "").strip().title()
        if not element or element == "H":
            continue
        atomic_number = periodic.GetAtomicNumber(element)
        if atomic_number <= 0:
            continue
        coords.append(np.asarray(atom.coord, dtype=float))
        atomic_numbers.append(atomic_number)
    if not coords:
        raise ValueError(f"No heavy atoms parsed from {path}")
    return np.asarray(coords), np.asarray(atomic_numbers, dtype=int)


def ligand_atoms(path: Path) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
    mol = read_first_sdf(path, remove_hs=False)
    conformer = mol.GetConformer()
    coords, atomic_numbers = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([point.x, point.y, point.z])
        atomic_numbers.append(atom.GetAtomicNum())
    return mol, np.asarray(coords, dtype=float), np.asarray(atomic_numbers, dtype=int)


def evaluate_pose(
    ligand_path: Path,
    protein_coords: np.ndarray,
    protein_elements: np.ndarray,
    pocket_coords: np.ndarray,
    reference_center: np.ndarray,
    *,
    tolerance: float,
    severe_threshold: float,
    pocket_cutoff: float,
) -> dict:
    mol, ligand_coords, ligand_elements = ligand_atoms(ligand_path)
    differences = ligand_coords[:, None, :] - protein_coords[None, :, :]
    distances = np.linalg.norm(differences, axis=-1).clip(min=1e-8)
    ligand_radii = np.asarray(
        [VDW_RADII.get(int(number), VDW_RADII[6]) for number in ligand_elements]
    )
    protein_radii = np.asarray(
        [VDW_RADII.get(int(number), VDW_RADII[6]) for number in protein_elements]
    )
    radius_sum = ligand_radii[:, None] + protein_radii[None, :]
    overlap = np.maximum(radius_sum - tolerance - distances, 0.0)
    positive = overlap[overlap > 0.0]

    pocket_distances = np.linalg.norm(
        ligand_coords[:, None, :] - pocket_coords[None, :, :], axis=-1
    )
    nearest_pocket = pocket_distances.min(axis=1)
    ligand_center = ligand_coords.mean(axis=0)

    return {
        "structure_path": str(ligand_path),
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True),
        "heavy_atom_count": int(len(ligand_coords)),
        "minimum_protein_ligand_distance": float(distances.min()),
        "vdw_clash_count": int(positive.size),
        "vdw_overlap_max": float(positive.max()) if positive.size else 0.0,
        "vdw_overlap_mean": float(positive.mean()) if positive.size else 0.0,
        "minimum_relative_distance": float(np.min(distances / radius_sum)),
        "severe_vdw_clash_count": int(np.count_nonzero(overlap > severe_threshold)),
        "pocket_occupancy_fraction": float(np.mean(nearest_pocket <= pocket_cutoff)),
        "minimum_pocket_atom_distance": float(nearest_pocket.min()),
        "centroid_distance_to_crossdocked_reference": float(
            np.linalg.norm(ligand_center - reference_center)
        ),
    }


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    args = parser.parse_args()
    config = load_config()
    directory = target_dir(config, args.target_id)
    manifest_path = directory / "properties" / "case_pose_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    protein_coords, protein_elements = structure_atoms(
        directory / "reference" / "pocket.pdb"
    )
    pocket_coords, _ = structure_atoms(directory / "reference" / "pocket.pdb")
    _, reference_coords, _ = ligand_atoms(
        directory / "reference" / "crossdocked_reference.sdf"
    )
    reference_center = reference_coords.mean(axis=0)

    rows = []
    for _, item in manifest.iterrows():
        common = {
            "target_id": int(item["target_id"]),
            "source_sample_idx": int(item["source_sample_idx"]),
            "stem": item["stem"],
        }
        pose_columns = [
            ("generated", "generated_sdf"),
            ("docked", "docked_sdf"),
        ]
        if "local_docked_sdf" in manifest and pd.notna(item.get("local_docked_sdf")):
            pose_columns.append(("local_docked", "local_docked_sdf"))
        for pose_kind, column in pose_columns:
            metrics = evaluate_pose(
                Path(item[column]),
                protein_coords,
                protein_elements,
                pocket_coords,
                reference_center,
                tolerance=float(config["vdw_tolerance"]),
                severe_threshold=float(config["severe_overlap_threshold"]),
                pocket_cutoff=float(config["pocket_contact_cutoff"]),
            )
            rows.append({**common, "pose_kind": pose_kind, **metrics})

    for reference_name, reference_file in (
        ("crossdocked_reference", directory / "reference" / "crossdocked_reference.sdf"),
        ("native_MA9", directory / "reference" / "native_MA9.sdf"),
    ):
        if reference_file.exists():
            metrics = evaluate_pose(
                reference_file,
                protein_coords,
                protein_elements,
                pocket_coords,
                reference_center,
                tolerance=float(config["vdw_tolerance"]),
                severe_threshold=float(config["severe_overlap_threshold"]),
                pocket_cutoff=float(config["pocket_contact_cutoff"]),
            )
            rows.append(
                {
                    "target_id": args.target_id,
                    "source_sample_idx": np.nan,
                    "stem": reference_name,
                    "pose_kind": "reference",
                    **metrics,
                }
            )

    long_frame = pd.DataFrame(rows)
    long_path = directory / "properties" / "case_clashes_long.csv"
    long_frame.to_csv(long_path, index=False)

    generated = long_frame[long_frame["pose_kind"].eq("generated")].drop(
        columns=["pose_kind"]
    )
    docked = long_frame[long_frame["pose_kind"].eq("docked")].drop(
        columns=["pose_kind", "target_id", "stem"]
    )
    docked = docked.rename(
        columns={
            column: f"{column}_docked"
            for column in docked.columns
            if column != "source_sample_idx"
        }
    )
    wide = generated.merge(docked, on="source_sample_idx", how="inner")
    if long_frame["pose_kind"].eq("local_docked").any():
        local_docked = long_frame[
            long_frame["pose_kind"].eq("local_docked")
        ].drop(columns=["pose_kind", "target_id", "stem"])
        local_docked = local_docked.rename(
            columns={
                column: f"{column}_local_docked"
                for column in local_docked.columns
                if column != "source_sample_idx"
            }
        )
        wide = wide.merge(local_docked, on="source_sample_idx", how="inner")
    wide_path = directory / "properties" / "case_clashes.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Wrote {len(long_frame)} pose records: {long_path}")
    print(f"Wrote {len(wide)} molecule records: {wide_path}")


if __name__ == "__main__":
    main()
