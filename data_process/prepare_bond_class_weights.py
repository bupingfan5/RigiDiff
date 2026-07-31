import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets import CrossDockDataset


def main():
    parser = argparse.ArgumentParser(
        description="Estimate fragment-bond class weights from the CrossDock train split."
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset_prefix", default="crossdocked_pocket10")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default="data/crossdocked_pocket10_bond_class_weights.json",
    )
    args = parser.parse_args()

    dataset = CrossDockDataset(
        data_path=args.data_dir,
        prefix=args.dataset_prefix,
        split="train",
        device="cpu",
    )
    count = len(dataset)
    if args.max_samples is not None:
        count = min(count, max(0, args.max_samples))

    counts = np.zeros(5, dtype=np.int64)
    failed = 0
    for index in tqdm(range(count), desc="Counting train bond classes"):
        try:
            bonds = dataset[index]["true_bonds"].cpu().numpy()
        except Exception:
            failed += 1
            continue
        upper = bonds[np.triu_indices(bonds.shape[0], k=1)]
        counts += np.bincount(upper, minlength=5)[:5]

    total = int(counts.sum())
    if total == 0:
        raise RuntimeError("No train bond pairs were counted")
    frequencies = counts.astype(np.float64) / total
    present = counts > 0
    weights = np.zeros(5, dtype=np.float64)
    raw = 1.0 / np.sqrt(np.maximum(frequencies[present], 1e-12))
    raw = raw / raw.mean()
    weights[present] = np.clip(raw, 0.1, 5.0)

    payload = {
        "samples_requested": count,
        "samples_failed": failed,
        "counts": counts.tolist(),
        "frequencies": frequencies.tolist(),
        "weights": weights.tolist(),
        "method": "inverse_sqrt_frequency_present_mean_normalized_clipped_0.1_5.0",
        "undirected_upper_triangle_only": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
