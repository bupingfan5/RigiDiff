import argparse
from pathlib import Path

import torch
import lmdb


def main():
    parser = argparse.ArgumentParser(description="Create a fixed CrossDock validation split without touching test keys.")
    parser.add_argument("--input", default="data/crossdocked_pocket10_pose_split.pt", type=Path)
    parser.add_argument("--output", default="data/crossdocked_pocket10_v2_split.pt", type=Path)
    parser.add_argument("--val_pockets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb", default=None, type=Path,
                        help="Optional LMDB used to pre-audit train keys before drawing validation pockets")
    parser.add_argument("--exclude_test_keys", default="",
                        help="Comma-separated exact test LMDB keys to exclude explicitly")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split: {args.output}")
    split = torch.load(args.input, map_location="cpu")
    train = torch.as_tensor(split["train"], dtype=torch.long)
    test = torch.as_tensor(split["test"], dtype=torch.long)
    excluded_test_keys = {
        int(value.strip()) for value in args.exclude_test_keys.split(",") if value.strip()
    }
    if excluded_test_keys:
        existing_test_keys = set(test.tolist())
        unknown = sorted(excluded_test_keys - existing_test_keys)
        if unknown:
            raise ValueError(f"Requested excluded keys are not in test split: {unknown}")
        test = torch.as_tensor(
            [int(key) for key in test.tolist() if int(key) not in excluded_test_keys],
            dtype=torch.long,
        )
    excluded_missing = 0
    if args.lmdb is not None:
        env = lmdb.open(str(args.lmdb), subdir=False, readonly=True, lock=False, readahead=False)
        with env.begin(write=False) as txn:
            available = {int(key.decode()) for key, _ in txn.cursor()}
        env.close()
        filtered = [int(key) for key in train.tolist() if int(key) in available]
        excluded_missing = len(train) - len(filtered)
        train = torch.as_tensor(filtered, dtype=torch.long)
    if len(set(train.tolist()) & set(test.tolist())):
        raise RuntimeError("Input train and test splits overlap")
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train), generator=generator)
    val_indices = permutation[:args.val_pockets]
    keep_indices = permutation[args.val_pockets:]
    output = {
        "train": train[keep_indices],
        "val": train[val_indices],
        "test": test.clone(),
        "metadata": {
            "validation_source": "train",
            "validation_seed": args.seed,
            "validation_pockets": args.val_pockets,
            "test_unchanged": not bool(excluded_test_keys),
            "excluded_test_keys": sorted(excluded_test_keys),
            "train_keys_excluded_as_missing": excluded_missing,
        },
    }
    if set(output["val"].tolist()) & set(output["test"].tolist()):
        raise RuntimeError("Generated validation split overlaps test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"Saved {args.output}: train={len(output['train'])}, val={len(output['val'])}, "
        f"test={len(output['test'])}"
    )


if __name__ == "__main__":
    main()
