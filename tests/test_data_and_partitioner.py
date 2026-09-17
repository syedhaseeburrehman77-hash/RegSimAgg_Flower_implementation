"""Unit tests for synthetic data generation, loader, and partitioner."""

from pathlib import Path
import pytest
import pandas as pd
import torch

from regsimagg.data import generate_synthetic_fets_data, make_loader
from regsimagg.partitioner import parse_fets_partition_csv, make_partitions


def test_synthetic_data_generation(tmp_path):
    out_dir = tmp_path / "synthetic_fets"
    generate_synthetic_fets_data(out_dir, num_clients=2, samples_per_client=1, volume_shape=(16, 16, 16))

    # Check directories
    c0_imgs = list((out_dir / "client_0" / "imagesTr").glob("*.nii.gz"))
    c0_lbls = list((out_dir / "client_0" / "labelsTr").glob("*.nii.gz"))
    assert len(c0_imgs) == 1
    assert len(c0_lbls) == 1

    # Check loader
    loader = make_loader(out_dir / "client_0", batch_size=1, patch_size=(16, 16, 16))
    for x, y in loader:
        assert x.shape == (1, 4, 16, 16, 16)
        assert y.shape == (1, 3, 16, 16, 16)
        break


def test_partitioner(tmp_path):
    csv_file = tmp_path / "partitioning.csv"
    df = pd.DataFrame({
        "Subject_ID": ["FeTS22_001", "FeTS22_002"],
        "Partition": [0, 1]
    })
    df.to_csv(csv_file, index=False)

    mapping = parse_fets_partition_csv(csv_file)
    assert mapping["FeTS22_001"] == "0"
    assert mapping["FeTS22_002"] == "1"
