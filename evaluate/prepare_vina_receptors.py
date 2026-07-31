import argparse
import csv
import pickle
from pathlib import Path

import lmdb

from vina_eval import (
    find_executable,
    prepare_receptor_meeko,
    reference_box_from_ligand,
    resolve_path,
    valid_split_keys,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare one immutable Meeko receptor set for Vina comparison.")
    parser.add_argument("--data_root", default="data/crossdocked_pocket10")
    parser.add_argument("--lmdb", default="data/crossdocked_pocket10_v2.lmdb")
    parser.add_argument("--split_file", default="data/crossdocked_pocket10_pose_split.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="work_vina_receptors/receptors")
    parser.add_argument("--manifest_csv", default="work_vina_receptors/receptor_manifest.csv")
    parser.add_argument("--box_margin", type=float, default=10.0)
    parser.add_argument("--box_min_size", type=float, default=20.0)
    parser.add_argument("--default_altloc", default="A")
    parser.add_argument("--allow_bad_res", dest="allow_bad_res", action="store_true", default=True)
    parser.add_argument("--no_allow_bad_res", dest="allow_bad_res", action="store_false")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    lmdb_path = resolve_path(args.lmdb)
    split_file = resolve_path(args.split_file)
    output_dir = resolve_path(args.output_dir)
    manifest_csv = resolve_path(args.manifest_csv)
    log_dir = output_dir.parent / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    receptor_tool = find_executable(["mk_prepare_receptor.py", "mk_prepare_receptor"])
    if receptor_tool is None:
        raise RuntimeError("Meeko mk_prepare_receptor is not available in PATH")

    keys = valid_split_keys(split_file, lmdb_path, args.split)
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    rows = []
    for target_id, key in enumerate(keys):
        row = {"target_id": target_id, "lmdb_key": key.decode(), "prep_success": False, "error": ""}
        try:
            with env.begin(write=False) as txn:
                sample = pickle.loads(txn.get(key))
            protein_path = data_root / sample["protein_filename"]
            ligand_path = data_root / sample["ligand_filename"]
            center, size = reference_box_from_ligand(
                ligand_path, args.box_margin, min_size=args.box_min_size
            )
            receptor_path = output_dir / f"target_{target_id}.pdbqt"
            log_path = log_dir / f"target_{target_id}_receptor_prep.log"
            prepare_receptor_meeko(
                protein_path,
                receptor_path,
                receptor_tool,
                args.default_altloc,
                allow_bad_res=args.allow_bad_res,
                log_path=log_path,
            )
            row.update({
                "protein_filename": sample["protein_filename"],
                "ligand_filename": sample["ligand_filename"],
                "receptor_pdbqt": str(receptor_path),
                "prep_log": str(log_path),
                "default_altloc": args.default_altloc,
                "allow_bad_res": args.allow_bad_res,
                "box_center_x": center[0], "box_center_y": center[1], "box_center_z": center[2],
                "box_size_x": size[0], "box_size_y": size[1], "box_size_z": size[2],
                "prep_success": True,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    env.close()

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(bool(row["prep_success"]) for row in rows)
    print(f"Receptor preparation: {success}/{len(rows)}; manifest={manifest_csv}")
    if success != len(rows):
        raise RuntimeError("Receptor preparation is incomplete; inspect the manifest and logs before evaluation")


if __name__ == "__main__":
    main()
