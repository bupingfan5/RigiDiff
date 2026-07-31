from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import pandas as pd

from case_utils import (
    CASE_ROOT,
    convert_pdbqt_to_sdf,
    crossdocked_reference_path,
    docked_pdbqt_path,
    generated_sdf_path,
    load_config,
    load_vina,
    project_path,
    sample_stem,
)


def selected_target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def prepare_target(target_id: int) -> Path:
    config = load_config()
    target_dir = selected_target_dir(config, target_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "properties").mkdir(exist_ok=True)
    (target_dir / "reference").mkdir(exist_ok=True)
    (target_dir / "poses" / "generated").mkdir(parents=True, exist_ok=True)
    (target_dir / "poses" / "docked").mkdir(parents=True, exist_ok=True)

    vina = load_vina(config)
    rows = vina[vina["target_id"].eq(target_id) & vina["vina"].notna()].copy()
    if rows.empty:
        raise RuntimeError(f"No successful Vina rows for target {target_id}")

    first = rows.iloc[0]
    pocket_rel = first["protein_filename"]
    pocket = project_path(config, "data_root") / pocket_rel
    reference = crossdocked_reference_path(config, pocket_rel)
    full_protein = project_path(config, "data_root") / Path(pocket_rel).parent / f"{first['pdb_id']}_full.pdb"
    prepared_protein = Path(str(first["prepared_protein_filename"]).replace("/mnt/e/fragdiff1/fragdiff", str(config["project_root"])))

    for source, name in (
        (pocket, "pocket.pdb"),
        (reference, "crossdocked_reference.sdf"),
        (full_protein, f"{str(first['pdb_id']).upper()}_full.pdb"),
        (prepared_protein, "prepared_protein.pdb"),
    ):
        if source.exists():
            destination = target_dir / "reference" / name
            destination.write_bytes(source.read_bytes())

    if target_id == 48:
        native_path = target_dir / "reference" / "native_MA9.sdf"
        if not native_path.exists():
            try:
                with urllib.request.urlopen(
                    config["native_ligand_url"], timeout=30
                ) as response:
                    native_path.write_bytes(response.read())
            except Exception as exc:
                print(
                    "Warning: optional 2OQV MA9 background ligand could not be "
                    f"downloaded ({exc}). The coordinate-matched reference 524 "
                    "is already available and the case-study pipeline can continue."
                )

    manifest_rows = []
    for _, row in rows.iterrows():
        stem = sample_stem(row)
        generated_source = generated_sdf_path(config, stem)
        docked_source = docked_pdbqt_path(config, stem)
        generated_dest = target_dir / "poses" / "generated" / f"{stem}_generated.sdf"
        docked_pdbqt_dest = target_dir / "poses" / "docked" / f"{stem}_docked.pdbqt"
        docked_sdf_dest = target_dir / "poses" / "docked" / f"{stem}_docked.sdf"

        if not generated_dest.exists():
            generated_dest.write_bytes(generated_source.read_bytes())
        if not docked_pdbqt_dest.exists():
            docked_pdbqt_dest.write_bytes(docked_source.read_bytes())
        if not docked_sdf_dest.exists():
            convert_pdbqt_to_sdf(config, docked_source, docked_sdf_dest)

        manifest_rows.append(
            {
                "target_id": target_id,
                "source_sample_idx": int(row["source_sample_idx"]),
                "stem": stem,
                "generated_sdf": str(generated_dest),
                "docked_pdbqt": str(docked_pdbqt_dest),
                "docked_sdf": str(docked_sdf_dest),
                "vina": row["vina"],
                "pocket_pdb": str(target_dir / "reference" / "pocket.pdb"),
                "prepared_protein_pdb": str(target_dir / "reference" / "prepared_protein.pdb"),
                "crossdocked_reference_sdf": str(target_dir / "reference" / "crossdocked_reference.sdf"),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    output = target_dir / "properties" / "case_pose_manifest.csv"
    manifest.to_csv(output, index=False)
    print(f"Prepared {len(manifest)} generated and docked poses: {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    args = parser.parse_args()
    prepare_target(args.target_id)


if __name__ == "__main__":
    main()
