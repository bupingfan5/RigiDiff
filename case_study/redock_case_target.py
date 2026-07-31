from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from case_utils import (
    CASE_ROOT,
    convert_pdbqt_to_sdf,
    load_config,
    project_path,
    read_first_sdf,
)


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def ligand_box(path: Path, padding: float, minimum: float):
    mol = read_first_sdf(path, remove_hs=False)
    conformer = mol.GetConformer()
    coords = np.asarray(
        [
            list(conformer.GetAtomPosition(atom.GetIdx()))
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1
        ],
        dtype=float,
    )
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    size = np.maximum(coords.max(axis=0) - coords.min(axis=0) + 2.0 * padding, minimum)
    return center, size


def score_from_pdbqt(path: Path) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1))
    raise ValueError(f"No Vina score found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config()
    directory = target_dir(config, args.target_id)
    manifest_path = directory / "properties" / "case_pose_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    local_dir = directory / "poses" / "local_docked"
    log_dir = directory / "logs" / "local_docking"
    local_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    center, size = ligand_box(
        directory / "reference" / "crossdocked_reference.sdf",
        float(config["local_box_padding"]),
        float(config["local_box_min_size"]),
    )
    receptor = directory / "reference" / "pocket_receptor.pdbqt"
    receptor_log = directory / "logs" / "pocket_receptor_openbabel.log"
    if args.overwrite:
        receptor.unlink(missing_ok=True)
    if not receptor.exists():
        command = [
            str(Path(config["obabel"])),
            str(directory / "reference" / "pocket.pdb"),
            "-O",
            str(receptor),
            "-xr",
            "-p",
            "7.4",
            "--partialcharge",
            "gasteiger",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        receptor_log.parent.mkdir(parents=True, exist_ok=True)
        receptor_log.write_text(
            completed.stdout + "\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not receptor.exists():
            raise RuntimeError(
                f"Pocket receptor preparation failed; see {receptor_log}"
            )
    vina = project_path(config, "vina_executable")
    if not vina.exists():
        raise FileNotFoundError(
            f"AutoDock Vina executable is missing: {vina}. "
            "Download the official v1.2.3 Windows binary as documented in README.md."
        )

    def dock(row: dict) -> dict:
        stem = row["stem"]
        ligand = (
            project_path(config, "vina_work_dir") / "ligands_pdbqt" / f"{stem}.pdbqt"
        )
        output = local_dir / f"{stem}_local_dock.pdbqt"
        output_sdf = local_dir / f"{stem}_local_dock.sdf"
        log = log_dir / f"{stem}.log"
        if args.overwrite:
            output.unlink(missing_ok=True)
            output_sdf.unlink(missing_ok=True)
        command = [
            str(vina),
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(size[0]),
            "--size_y",
            str(size[1]),
            "--size_z",
            str(size[2]),
            "--exhaustiveness",
            str(config["local_docking_exhaustiveness"]),
            "--num_modes",
            str(config["local_docking_num_modes"]),
            "--seed",
            str(config["local_docking_seed"]),
            "--cpu",
            "1",
            "--out",
            str(output),
        ]
        if not output.exists():
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            log.write_text(
                completed.stdout + "\n" + completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Vina failed for {stem}; see {log}")
        convert_pdbqt_to_sdf(config, output, output_sdf)
        return {
            "target_id": int(row["target_id"]),
            "source_sample_idx": int(row["source_sample_idx"]),
            "stem": stem,
            "local_vina": score_from_pdbqt(output),
            "local_docked_pdbqt": str(output),
            "local_docked_sdf": str(output_sdf),
            "local_docking_log": str(log),
        }

    records = []
    with ThreadPoolExecutor(max_workers=int(config["local_docking_workers"])) as pool:
        futures = [pool.submit(dock, row) for row in manifest.to_dict("records")]
        for future in as_completed(futures):
            records.append(future.result())
    results = pd.DataFrame(records).sort_values("source_sample_idx")
    output_csv = directory / "properties" / "case_local_docking.csv"
    results.to_csv(output_csv, index=False)

    updated = manifest.drop(
        columns=[
            "local_vina",
            "local_docked_pdbqt",
            "local_docked_sdf",
            "local_docking_log",
        ],
        errors="ignore",
    ).merge(results, on=["target_id", "source_sample_idx", "stem"], how="left")
    updated.to_csv(manifest_path, index=False)
    metadata = {
        "target_id": args.target_id,
        "receptor": str(receptor),
        "reference_ligand": str(
            directory / "reference" / "crossdocked_reference.sdf"
        ),
        "box_center": center.tolist(),
        "box_size": size.tolist(),
        "seed": int(config["local_docking_seed"]),
        "exhaustiveness": int(config["local_docking_exhaustiveness"]),
        "num_modes": int(config["local_docking_num_modes"]),
        "vina_version": "1.2.3",
        "purpose": (
            "Case-study-only local redocking against the exact cropped pocket "
            "provided to the generation model. Main-table full-protein scores "
            "are retained separately and are not overwritten."
        ),
    }
    (directory / "properties" / "case_local_docking_protocol.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Local redocking complete for {len(results)} candidates; "
        f"median={results['local_vina'].median():.3f} kcal/mol"
    )
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
