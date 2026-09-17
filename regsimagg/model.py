"""Model architectures and segmentation evaluation utilities.

Integrates MONAI 3D U-Net with FeTS/BraTS subregion conversions:
- Whole Tumor (WT): labels 1, 2, 4
- Tumor Core (TC): labels 1, 4
- Enhancing Tumor (ET): label 4
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.losses import DiceLoss, DiceCELoss
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# ---------------------------------------------------------------------------
# FeTS / BraTS Subregion Mapping
# ---------------------------------------------------------------------------

def convert_labels_to_subregions(labels: torch.Tensor) -> torch.Tensor:
    """Convert multi-class segmentation labels into 3 binary subregion channels.

    BraTS standard:
      - WT (Whole Tumor): labels 1, 2, 4
      - TC (Tumor Core):  labels 1, 4
      - ET (Enhancing Tumor): label 4

    Args:
        labels: Tensor of shape (B, 1, D, H, W) or (B, D, H, W) containing integers {0, 1, 2, 4}.
    Returns:
        Tensor of shape (B, 3, D, H, W) where channels correspond to [WT, TC, ET].
    """
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)

    wt = (labels == 1) | (labels == 2) | (labels == 4)
    tc = (labels == 1) | (labels == 4)
    et = (labels == 4)

    return torch.cat([wt, tc, et], dim=1).float()


def convert_subregion_logits_to_labels(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert 3 subregion sigmoid probabilities [WT, TC, ET] back to BraTS labels {0, 1, 2, 4}.

    Precedence:
      1. If ET > threshold -> label 4
      2. Else if TC > threshold -> label 1
      3. Else if WT > threshold -> label 2
      4. Else -> label 0 (background)
    """
    probs = torch.sigmoid(logits)
    wt = probs[:, 0] > threshold
    tc = probs[:, 1] > threshold
    et = probs[:, 2] > threshold

    out = torch.zeros_like(wt, dtype=torch.long)
    out[wt] = 2
    out[tc] = 1
    out[et] = 4
    return out


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

class Small3DUNet(nn.Module):
    """Lightweight 3D U-Net for rapid experimentation and testing."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 8):
        super().__init__()
        self.enc1 = self._block(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._block(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = self._block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)
        self.dec1 = self._block(base * 2, base)
        self.out = nn.Conv3d(base, out_channels, 1)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def build_fets_unet(
    in_channels: int = 4,
    out_channels: int = 3,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
) -> nn.Module:
    """Instantiate the standard FeTS 3D U-Net using MONAI."""
    if HAS_MONAI:
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.INSTANCE,
        )
    else:
        logger.warning("MONAI not installed, falling back to Small3DUNet.")
        return Small3DUNet(in_channels=in_channels, out_channels=out_channels, base=16)


def get_model(model_type: str = "monai_unet", in_channels: int = 4, out_channels: int = 3) -> nn.Module:
    """Factory function for model creation."""
    if model_type == "monai_unet" and HAS_MONAI:
        return build_fets_unet(in_channels=in_channels, out_channels=out_channels)
    return Small3DUNet(in_channels=in_channels, out_channels=out_channels)


# ---------------------------------------------------------------------------
# Loss Functions & Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5) -> float:
    """Binary Dice score over spatial dimensions."""
    preds_flat = preds.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    intersection = (preds_flat * targets_flat).sum()
    denominator = preds_flat.sum() + targets_flat.sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


def evaluate_subregions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    include_hausdorff: bool = False,
) -> Dict[str, float]:
    """Compute FeTS challenge evaluation metrics across subregions.

    Args:
        logits: Model predictions of shape (B, 3, D, H, W)
        targets: Ground truth subregions of shape (B, 3, D, H, W) or labels of shape (B, 1, D, H, W)
    Returns:
        Dictionary of Dice (and optionally Hausdorff) metrics for WT, TC, ET.
    """
    if targets.shape[1] == 1:
        targets = convert_labels_to_subregions(targets)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    dice_wt = compute_dice_score(preds[:, 0], targets[:, 0])
    dice_tc = compute_dice_score(preds[:, 1], targets[:, 1])
    dice_et = compute_dice_score(preds[:, 2], targets[:, 2])
    dice_avg = (dice_wt + dice_tc + dice_et) / 3.0

    metrics = {
        "dice_wt": dice_wt,
        "dice_tc": dice_tc,
        "dice_et": dice_et,
        "dice_avg": dice_avg,
    }

    if include_hausdorff and HAS_MONAI:
        try:
            hd_metric = HausdorffDistanceMetric(percentile=95, include_background=True)
            hd_metric(y_pred=preds, y=targets)
            hd_values = hd_metric.aggregate()
            if isinstance(hd_values, torch.Tensor):
                hd_arr = hd_values.cpu().numpy()
                metrics["hd95_wt"] = float(hd_arr[0]) if len(hd_arr) > 0 else 0.0
                metrics["hd95_tc"] = float(hd_arr[1]) if len(hd_arr) > 1 else 0.0
                metrics["hd95_et"] = float(hd_arr[2]) if len(hd_arr) > 2 else 0.0
        except Exception as e:
            logger.warning("Hausdorff metric computation failed: %s", e)

    return metrics
