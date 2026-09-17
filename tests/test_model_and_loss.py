"""Unit tests for MONAI model, subregion conversions, and evaluation metrics."""

import pytest
import torch
from regsimagg.model import (
    Small3DUNet,
    convert_labels_to_subregions,
    convert_subregion_logits_to_labels,
    evaluate_subregions,
)


def test_subregion_conversions():
    # Labels shape (1, 1, 4, 4, 4) with labels {0, 1, 2, 4}
    labels = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    labels[0, 0, 1, 1, 1] = 1  # NCR (in TC, WT)
    labels[0, 0, 2, 2, 2] = 2  # ED (in WT only)
    labels[0, 0, 3, 3, 3] = 4  # ET (in ET, TC, WT)

    subregions = convert_labels_to_subregions(labels)
    assert subregions.shape == (1, 3, 4, 4, 4)

    # WT: index 0 -> should include 1, 2, 4
    assert subregions[0, 0, 1, 1, 1] == 1.0
    assert subregions[0, 0, 2, 2, 2] == 1.0
    assert subregions[0, 0, 3, 3, 3] == 1.0

    # TC: index 1 -> should include 1 and 4
    assert subregions[0, 1, 1, 1, 1] == 1.0
    assert subregions[0, 1, 2, 2, 2] == 0.0
    assert subregions[0, 1, 3, 3, 3] == 1.0

    # ET: index 2 -> should include 4 only
    assert subregions[0, 2, 1, 1, 1] == 0.0
    assert subregions[0, 2, 2, 2, 2] == 0.0
    assert subregions[0, 2, 3, 3, 3] == 1.0


def test_model_forward():
    model = Small3DUNet(in_channels=4, out_channels=3, base=4)
    x = torch.randn(1, 4, 16, 16, 16)
    out = model(x)
    assert out.shape == (1, 3, 16, 16, 16)


def test_evaluate_subregions():
    logits = torch.zeros((1, 3, 8, 8, 8))
    targets = torch.zeros((1, 3, 8, 8, 8))
    # Perfect overlap
    logits[0, :, 2:6, 2:6, 2:6] = 10.0  # sigmoid -> ~1.0
    targets[0, :, 2:6, 2:6, 2:6] = 1.0

    metrics = evaluate_subregions(logits, targets)
    assert metrics["dice_wt"] > 0.95
    assert metrics["dice_tc"] > 0.95
    assert metrics["dice_et"] > 0.95
    assert metrics["dice_avg"] > 0.95
