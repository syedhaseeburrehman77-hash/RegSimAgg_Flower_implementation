"""Unit tests for inference and model outputs to disc."""

from pathlib import Path
import pytest
import torch

from regsimagg.model import Small3DUNet
from regsimagg.data import generate_synthetic_fets_data
from regsimagg.inference import model_outputs_to_disc


def test_model_outputs_to_disc(tmp_path):
    data_dir = tmp_path / "data"
    generate_synthetic_fets_data(data_dir, num_clients=1, samples_per_client=1, volume_shape=(16, 16, 16))

    # Save a small model checkpoint
    model = Small3DUNet(base=4)
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), str(model_path))

    out_dir = tmp_path / "predictions"
    model_outputs_to_disc(
        data_path=data_dir / "client_0" / "imagesTr",
        output_path=out_dir,
        native_model_path=model_path,
        roi_size=(16, 16, 16),
    )

    outputs = list(out_dir.glob("*.nii*"))
    assert len(outputs) >= 1
