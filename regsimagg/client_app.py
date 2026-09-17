"""Flower ClientApp for FeTS / BraTS 3D tumor segmentation.

Supports:
1. Dynamic hyperparameter execution (learning rate, epochs, fixed batches).
2. MONAI 3D U-Net and Dice loss.
3. Subregion segmentation metrics (WT, TC, ET Dice).
"""

from __future__ import annotations
import logging
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
from flwr.client import ClientApp
from flwr.app import ArrayRecord, MetricRecord, RecordDict, Message

from .model import get_model, evaluate_subregions
from .data import make_loader

try:
    from monai.losses import DiceLoss
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

logger = logging.getLogger(__name__)
app = ClientApp()


class MultiLabelDiceLoss(nn.Module):
    """Sigmoid-based multi-label Dice loss for WT, TC, ET."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Spatial dimensions: (D, H, W) -> (2, 3, 4)
        dims = (2, 3, 4)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        # Average across batches and channels
        return 1.0 - dice.mean()


def get_loss_fn() -> nn.Module:
    if HAS_MONAI:
        return DiceLoss(sigmoid=True)
    return MultiLabelDiceLoss()


@app.train()
def train(msg: Message, context) -> Message:
    """Execute local training round."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = str(context.run_config.get("model-type", "small_unet"))
    model = get_model(model_type=model_type).to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.train()

    root = context.run_config.get("data-root", "./data")
    pid = int(context.node_config.get("partition-id", context.node_id))
    part_csv = context.run_config.get("partition-csv", None)
    batch_size = int(context.run_config.get("batch-size", 1))
    num_workers = int(context.run_config.get("num-workers", 0))

    if part_csv and Path(part_csv).exists():
        loader = make_loader(
            data_root=root,
            partition_csv=part_csv,
            partition_id=pid,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        client_root = f"{root}/client_{pid}"
        loader = make_loader(client_dir=client_root, batch_size=batch_size, num_workers=num_workers)

    # Dynamic hyperparameters received from server
    lr = float(msg.content["config"].get("lr", context.run_config.get("learning-rate", 5e-5)))
    epochs_cfg = float(msg.content["config"].get("epochs", context.run_config.get("local-epochs", 1.0)))
    batches_cfg = int(msg.content["config"].get("batches", -1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_fn()

    total_loss = 0.0
    total_samples = 0

    if batches_cfg > 0:
        # FeTS fixed_number_of_batches mode: loop over dataset until batches completed
        batch_count = 0
        while batch_count < batches_cfg:
            for x, y in loader:
                if batch_count >= batches_cfg:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y.float())
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * x.size(0)
                total_samples += x.size(0)
                batch_count += 1
    else:
        # Epoch-based training
        num_epochs = max(1, int(round(epochs_cfg)))
        for _ in range(num_epochs):
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y.float())
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * x.size(0)
                total_samples += x.size(0)

    train_loss = total_loss / max(total_samples, 1)
    metrics = {
        "train_loss": train_loss,
        "num-examples": total_samples,
    }

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context) -> Message:
    """Execute local evaluation round."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = str(context.run_config.get("model-type", "small_unet"))
    model = get_model(model_type=model_type).to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.eval()

    root = context.run_config.get("data-root", "./data")
    pid = int(context.node_config.get("partition-id", context.node_id))
    part_csv = context.run_config.get("partition-csv", None)
    num_workers = int(context.run_config.get("num-workers", 0))

    if part_csv and Path(part_csv).exists():
        loader = make_loader(
            data_root=root,
            partition_csv=part_csv,
            partition_id=pid,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        client_root = f"{root}/client_{pid}"
        loader = make_loader(client_dir=client_root, batch_size=1, num_workers=num_workers, shuffle=False)

    loss_fn = get_loss_fn()
    total_loss = 0.0
    total_samples = 0
    all_dice_wt, all_dice_tc, all_dice_et = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y.float())
            total_loss += float(loss.item()) * x.size(0)
            total_samples += x.size(0)

            sub_metrics = evaluate_subregions(logits, y)
            all_dice_wt.append(sub_metrics["dice_wt"])
            all_dice_tc.append(sub_metrics["dice_tc"])
            all_dice_et.append(sub_metrics["dice_et"])

    eval_loss = total_loss / max(total_samples, 1)
    mean_dice_wt = float(sum(all_dice_wt) / max(len(all_dice_wt), 1))
    mean_dice_tc = float(sum(all_dice_tc) / max(len(all_dice_tc), 1))
    mean_dice_et = float(sum(all_dice_et) / max(len(all_dice_et), 1))
    mean_dice_avg = (mean_dice_wt + mean_dice_tc + mean_dice_et) / 3.0

    eval_metrics = {
        "eval_loss": eval_loss,
        "dice_wt": mean_dice_wt,
        "dice_tc": mean_dice_tc,
        "dice_et": mean_dice_et,
        "dice_avg": mean_dice_avg,
        "num-examples": total_samples,
    }

    return Message(
        content=RecordDict({
            "metrics": MetricRecord(eval_metrics),
        }),
        reply_to=msg,
    )
