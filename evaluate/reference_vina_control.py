import argparse
import csv
import pickle
import statistics
from pathlib import Path

import lmdb

from vina_eval import resolve_path, valid_split_keys


def prepare_manifest(args):
    data_root = resolve_path(args.data_root)
    lmdb_path = resolve_path(args.lmdb)
    keys = valid_split_keys(resolve_path(args.split_file), lmdb_path, args.split)
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    rows = []
    for target_id, key in enumerate(keys):
        with env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(key))
        rows.append({
            "sdf_path": str(data_root / sample["ligand_filename"]),
            "target_name": f"target_{target_id}",
            "dataset_target_index": target_id,
            "lmdb_key": key.decode(),
            "reference_control": True,
        })
    env.close()
    output = resolve_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reference control manifest: {output} ({len(rows)} targets)")


def check_results(args):
    path = resolve_path(args.results_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        try:
            values.append(float(row["vina_score"]))
        except (KeyError, TypeError, ValueError):
            pass
    success_rate = len(values) / len(rows) if rows else 0.0
    negative_rate = sum(value < 0 for value in values) / len(values) if values else 0.0
    score_median = statistics.median(values) if values else float("inf")
    print(
        f"Reference control: rows={len(rows)}, score_success={success_rate:.3%}, "
        f"negative={negative_rate:.3%}, median={score_median:.3f}"
    )
    if len(rows) != args.expected_targets or success_rate < 1.0 or negative_rate < 0.90 or score_median >= 0:
        raise RuntimeError(
            "Reference-ligand control failed. Stop model optimization and repair receptor/ligand preparation first."
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare or verify the 100-reference-ligand Vina control.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data_root", default="data/crossdocked_pocket10")
    prepare.add_argument("--lmdb", default="data/crossdocked_pocket10_v2.lmdb")
    prepare.add_argument("--split_file", default="data/crossdocked_pocket10_pose_split.pt")
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--output_csv", default="work_vina_reference/reference_manifest.csv")
    prepare.set_defaults(func=prepare_manifest)

    check = subparsers.add_parser("check")
    check.add_argument("--results_csv", default="work_vina_reference/reference_vina_score.csv")
    check.add_argument("--expected_targets", type=int, default=100)
    check.set_defaults(func=check_results)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
