import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.edm import isotropic_gaussian_so3
from src.value_net import ValueNetwork


def get_alphas_cumprod(timesteps=1000):
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    return alphas_cumprod / alphas_cumprod[0]


def _pad_rot(t: torch.Tensor, max_len: int):
    out = torch.eye(3, dtype=t.dtype, device=t.device).view(1, 3, 3).repeat(max_len, 1, 1)
    out[: t.shape[0]] = t
    return out


def get_fc_edges(node_mask):
    bsz, n_nodes, _ = node_mask.shape
    device = node_mask.device
    rows, cols = [], []
    for batch_idx in range(bsz):
        valid_idx = (node_mask[batch_idx, :, 0] > 0.5).nonzero().view(-1)
        if len(valid_idx) == 0:
            continue
        rows.append(valid_idx.repeat_interleave(len(valid_idx)) + batch_idx * n_nodes)
        cols.append(valid_idx.repeat(len(valid_idx)) + batch_idx * n_nodes)

    if len(rows) == 0:
        return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(rows), torch.cat(cols)


def load_value_dataset(data_path):
    data = torch.load(data_path, map_location="cpu")
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"{data_path} does not contain any value-network samples.")
    return data


def infer_feature_dim(data):
    dims = [int(item["one_hot"].shape[-1]) for item in data if item is not None and "one_hot" in item]
    if not dims:
        raise RuntimeError("Value-network dataset does not contain any one_hot features.")
    return max(dims)


def _pad_one_hot(one_hot: torch.Tensor, feature_dim: int):
    cur_dim = int(one_hot.shape[-1])
    if cur_dim > feature_dim:
        raise RuntimeError(f"one_hot feature dim {cur_dim} is larger than configured dim {feature_dim}.")
    one_hot = one_hot.float()
    if cur_dim == feature_dim:
        return one_hot
    return torch.cat([one_hot, one_hot.new_zeros(one_hot.shape[0], feature_dim - cur_dim)], dim=-1)


def collate_fn(batch, feature_dim=None):
    max_len = max(b["positions"].shape[0] for b in batch)
    if feature_dim is None:
        feature_dim = max(int(b["one_hot"].shape[-1]) for b in batch)

    x_pad, rot_pad, h_pad, masks, scores = [], [], [], [], []
    for sample in batch:
        n_nodes = sample["positions"].shape[0]
        pos = sample["positions"].float()
        x_pad.append(torch.cat([pos, pos.new_zeros(max_len - n_nodes, 3)]))

        one_hot = _pad_one_hot(sample["one_hot"], feature_dim)
        h_pad.append(torch.cat([one_hot, one_hot.new_zeros(max_len - n_nodes, feature_dim)]))

        if "frag_rotations" in sample:
            rot_pad.append(_pad_rot(sample["frag_rotations"].float(), max_len))
        elif "rotations" in sample:
            rot_pad.append(_pad_rot(sample["rotations"].float(), max_len))
        else:
            rot_pad.append(_pad_rot(torch.eye(3).repeat(n_nodes, 1, 1), max_len))

        mask = torch.zeros(max_len, 1)
        sample_mask = sample.get("node_mask")
        if sample_mask is None:
            mask[:n_nodes] = 1.0
        else:
            sample_mask = sample_mask.float().view(-1)
            mask[:n_nodes, 0] = sample_mask[:n_nodes]
        masks.append(mask)
        scores.append(float(sample["score"]))

    return (
        torch.stack(x_pad),
        torch.stack(rot_pad),
        torch.stack(h_pad),
        torch.stack(masks),
        torch.tensor(scores, dtype=torch.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fragment-level value network.")
    parser.add_argument("--data_path", type=str, default="data/value_network_dataset.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_path", type=str, default="checkpoints/value_net_final.pt")
    return parser.parse_args()


def epoch_checkpoint_path(output_path, epoch):
    directory = os.path.dirname(output_path) or "."
    stem, ext = os.path.splitext(os.path.basename(output_path))
    ext = ext or ".pt"
    if stem == "value_net_final":
        return os.path.join(directory, f"value_net_ep{epoch}{ext}")
    return os.path.join(directory, f"{stem}_ep{epoch}{ext}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading value-network dataset: {args.data_path}")
    data = load_value_dataset(args.data_path)
    feature_dim = infer_feature_dim(data)
    print(f"ValueNetwork input feature dim: {feature_dim}")

    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, feature_dim=feature_dim),
    )
    if len(dataloader) == 0:
        raise RuntimeError("Value-network dataloader is empty.")

    model = ValueNetwork(in_node_nf=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    alphas_cumprod = get_alphas_cumprod().to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    print("Training rigid-fragment value network...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x_0, rot_0, h, mask, score_true in pbar:
            x_0, rot_0 = x_0.to(device), rot_0.to(device)
            h, mask, score_true = h.to(device), mask.to(device), score_true.to(device)
            batch_size = x_0.size(0)

            optimizer.zero_grad()

            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            alpha_bar = alphas_cumprod[t].view(batch_size, 1, 1)

            noise_x = torch.randn_like(x_0)
            x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise_x
            rot_t, _ = isotropic_gaussian_so3(rot_0, t, sqrt_one_minus_alphas_cumprod)
            edge_index = get_fc_edges(mask)

            score_pred = model(x_t, rot_t, h, t, mask, edge_index).squeeze(-1)
            loss = criterion(score_pred, score_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), epoch_checkpoint_path(args.output_path, epoch + 1))

    torch.save(model.state_dict(), args.output_path)
    print(f"Value network saved to {args.output_path}")


if __name__ == "__main__":
    main()
