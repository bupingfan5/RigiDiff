"""Short mixed-precision smoke test for corrective-training numerical stability."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from src.datasets import CrossDockDataset, get_dataloader
from src.lightning import DDPM


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")
    torch.manual_seed(42)

    dataset = CrossDockDataset(
        data_path=str(root / "data"),
        prefix="crossdocked_pocket10",
        split="train",
        device="cpu",
    )
    loader = get_dataloader(dataset, batch_size=4, shuffle=True)
    model = DDPM.load_from_checkpoint(
        str(root / "checkpoints" / "run6" / "last.ckpt"),
        map_location="cpu",
        strict=False,
    ).to(device).train()
    model.lr = 1.0e-5
    model.pocket_encoder_lr_multiplier = 0.1
    model.bond_head_lr_multiplier = 1.0
    trainable_tokens = (
        "pocket_feature_encoder",
        "node_role_embedding",
        "edge_predictor",
        "attach_src_predictor",
        "attach_dst_predictor",
        "presence_predictor",
    )
    for name, parameter in model.edm.named_parameters():
        parameter.requires_grad = any(token in name for token in trainable_tokens)
    optimizer = model.configure_optimizers()

    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.forward(batch, training=True)
        values = {
            "loss": float(loss.detach()),
            "contact": float(metrics["loss_contact"].detach()),
            "clash": float(metrics["loss_pocket_clash"].detach()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            all_values = {name: float(value.detach()) for name, value in metrics.items()}
            raise RuntimeError(f"Non-finite smoke-test metric at step {step}: {all_values}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        bad_parameters = [
            name for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if bad_parameters:
            raise RuntimeError(f"Non-finite parameters after step {step}: {bad_parameters[:10]}")
        print(
            f"step={step:02d} loss={values['loss']:.4f} "
            f"contact={values['contact']:.4f} clash={values['clash']:.4f} "
            f"grad_norm={float(grad_norm):.4f}"
        )
        if step >= 30:
            break


if __name__ == "__main__":
    main()
