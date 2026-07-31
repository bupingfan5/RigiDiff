import argparse
import json
from pathlib import Path

from src.datasets import CrossDockDataset
from src.generation_utils import fragment_count_histogram


def main():
    parser = argparse.ArgumentParser(description="Audit the train-split BRICS fragment-count distribution.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset_prefix", default="crossdocked_pocket10_v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/crossdocked_pocket10_v2_fragment_count_hist.json")
    args = parser.parse_args()

    dataset = CrossDockDataset(
        data_path=args.data_dir,
        prefix=args.dataset_prefix,
        split="train",
        device="cpu",
    )
    histogram, failures = fragment_count_histogram(dataset, limit=args.limit)
    payload = {
        "dataset_prefix": args.dataset_prefix,
        "split": "train",
        "processed": sum(histogram.values()) + failures,
        "success": sum(histogram.values()),
        "failures": failures,
        "histogram": histogram,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
