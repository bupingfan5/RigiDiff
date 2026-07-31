import argparse
import json
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
from rdkit import RDLogger

from src.fragment_vocab import (
    _build_ligand_mol_from_sample,
    _fragment_instances_from_mol,
    _resolve_lmdb_path,
    fit_fixed_template_pose,
    get_fragment_vocab_path,
    load_fragment_vocabulary,
)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
        "rate_above_0_1": float((array > 0.1).mean()),
        "rate_above_0_5": float((array > 0.5).mean()),
        "rate_above_1_0": float((array > 1.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit fixed-vocabulary fragment pose labels against LMDB conformers."
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset_prefix", default="crossdocked_pocket10")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output_json", default="fragment_frame_audit.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    lmdb_path = _resolve_lmdb_path(args.data_root, args.dataset_prefix)
    vocab_path = get_fragment_vocab_path(args.data_root, args.dataset_prefix)
    vocabulary = load_fragment_vocabulary(vocab_path, expected_lmdb_path=lmdb_path)

    legacy_rmsd = []
    fixed_template_rmsd = []
    samples_seen = 0
    samples_failed = 0
    fragments_missing_from_vocab = 0

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )
    with env.begin(write=False) as txn:
        for _, value in txn.cursor():
            if samples_seen >= args.max_samples:
                break
            samples_seen += 1
            try:
                sample = pickle.loads(value)
                mol = _build_ligand_mol_from_sample(sample)
                if mol is None:
                    raise ValueError("ligand molecule construction failed")
                fragment_instances, _, _ = _fragment_instances_from_mol(mol)
                for fragment in fragment_instances:
                    type_idx = vocabulary["smiles_to_idx"].get(fragment["smiles"])
                    if type_idx is None:
                        fragments_missing_from_vocab += 1
                        continue
                    template = np.asarray(
                        vocabulary["entries"][int(type_idx)]["coords"], dtype=np.float64
                    )
                    legacy = np.asarray(fragment["local_coords"], dtype=np.float64)
                    if template.shape == legacy.shape:
                        legacy_rmsd.append(
                            float(np.sqrt(np.mean(np.sum((template - legacy) ** 2, axis=1))))
                        )
                    _, _, fitted_rmsd = fit_fixed_template_pose(
                        template,
                        fragment["global_coords"],
                    )
                    fixed_template_rmsd.append(fitted_rmsd)
            except Exception:
                samples_failed += 1
    env.close()

    fixed_summary = summarize(fixed_template_rmsd)
    thresholds = {
        "median_at_most": 0.05,
        "p95_at_most": 0.50,
        "rate_above_1_0_at_most": 0.01,
    }
    passed = bool(
        fixed_summary.get("count", 0) > 0
        and fixed_summary["median"] <= thresholds["median_at_most"]
        and fixed_summary["p95"] <= thresholds["p95_at_most"]
        and fixed_summary["rate_above_1_0"] <= thresholds["rate_above_1_0_at_most"]
    )
    result = {
        "lmdb": os.path.abspath(lmdb_path),
        "fragment_vocabulary": os.path.abspath(vocab_path),
        "samples_seen": samples_seen,
        "samples_failed": samples_failed,
        "fragments_missing_from_vocab": fragments_missing_from_vocab,
        "legacy_independent_pca_frame_rmsd": summarize(legacy_rmsd),
        "fixed_template_kabsch_rmsd": fixed_summary,
        "thresholds": thresholds,
        "passed": passed,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"FRAME_AUDIT={'PASSED' if passed else 'FAILED'}")
    if args.strict and not passed:
        raise RuntimeError("Fixed-template fragment frame audit did not meet thresholds")


if __name__ == "__main__":
    main()
