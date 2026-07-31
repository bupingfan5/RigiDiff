import argparse
import math
import os
import pickle
import sys

import torch
from rdkit import RDConfig
from rdkit.Chem import Descriptors, QED
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except ImportError:
    sascorer = None

from src.datasets import CrossDockDataset, PDBbindDataset, build_ligand_mol_from_sample


def calculate_composite_score(mol):
    """Return a 0-1 composite molecular quality score."""
    if sascorer is None:
        return 0.0

    try:
        qed = QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
    except Exception:
        return 0.0

    sa_score = max(0.0, min(1.0, 1.0 - (sa - 2.5) / 3.5))

    if 1.0 <= logp <= 3.0:
        logp_score = 1.0
    elif logp < 1.0:
        logp_score = math.exp(-2.0 * (1.0 - logp) ** 2)
    else:
        logp_score = math.exp(-2.0 * (logp - 3.0) ** 2)

    if 300 <= mw <= 500:
        mw_score = 1.0
    elif mw < 300:
        mw_score = math.exp(-0.0005 * (300 - mw) ** 2)
    else:
        mw_score = math.exp(-0.0005 * (mw - 500) ** 2)

    return (qed * 0.50) + (sa_score * 0.35) + (logp_score * 0.10) + (mw_score * 0.05)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fragment-level value-network training data.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_prefix", type=str, default="PDBbind_pocket10")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=-1, help="Maximum number of dataset indices to scan. <=0 means all.")
    return parser.parse_args()


def resolve_output_path(data_dir, dataset_prefix, output):
    if output:
        return output
    if dataset_prefix == "PDBbind_pocket10":
        return os.path.join(data_dir, "value_network_dataset.pt")
    safe_prefix = dataset_prefix.replace("/", "_").replace("\\", "_")
    return os.path.join(data_dir, f"value_network_dataset_{safe_prefix}.pt")


def select_dataset_class(dataset_prefix):
    if "crossdock" in dataset_prefix.lower():
        return CrossDockDataset
    return PDBbindDataset


def get_sample_for_item(dataset, item, fallback_idx):
    key = item.get("uuid") if isinstance(item, dict) else None
    if key is None:
        key = dataset.valid_keys[fallback_idx]
    elif isinstance(key, str):
        key = key.encode()
    elif not isinstance(key, bytes):
        key = str(key).encode()

    with dataset.env.begin(write=False) as txn:
        value = txn.get(key)
    if value is None:
        return None
    return pickle.loads(value)


def clone_cpu_tensor(value):
    return value.detach().clone().cpu()


def main():
    args = parse_args()
    output_path = resolve_output_path(args.data_dir, args.dataset_prefix, args.output)
    dataset_cls = select_dataset_class(args.dataset_prefix)

    print(f"Initializing {dataset_cls.__name__}: prefix={args.dataset_prefix}, split={args.split}")
    dataset = dataset_cls(data_path=args.data_dir, prefix=args.dataset_prefix, split=args.split)

    n_scan = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    value_data_list = []

    print("Extracting fragment features and molecular scores...")
    for idx in tqdm(range(n_scan)):
        try:
            item = dataset[idx]
            if item is None:
                continue

            sample = get_sample_for_item(dataset, item, idx)
            if sample is None:
                continue

            mol = build_ligand_mol_from_sample(sample)
            if mol is None:
                continue

            score = calculate_composite_score(mol)
            if score == 0.0:
                continue

            value_data_list.append(
                {
                    "positions": clone_cpu_tensor(item["positions"]),
                    "frag_rotations": clone_cpu_tensor(item["frag_rotations"]),
                    "one_hot": clone_cpu_tensor(item["one_hot"]),
                    "node_mask": clone_cpu_tensor(item["fragment_mask"]),
                    "score": float(score),
                }
            )
        except Exception:
            continue

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(value_data_list, output_path)
    print(f"Saved {len(value_data_list)} value-network samples to {output_path}")


if __name__ == "__main__":
    main()
