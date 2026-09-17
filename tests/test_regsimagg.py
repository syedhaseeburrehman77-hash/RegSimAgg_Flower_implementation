"""Unit tests for RegSimAgg and baseline aggregation algorithms."""

import pytest
import numpy as np
import torch
from collections import OrderedDict

from regsimagg.regsimagg import (
    my_sum,
    paper_l1_distance,
    github_compat_distance,
    similarity_weights,
    sample_weights,
    base_weights,
    regularize_weights,
    reg_sim_weighted_average_layerwise,
    aggregate_states,
    fedavg_aggregate,
    clipped_aggregate,
    fedavgm_aggregate,
)


def test_my_sum():
    # 1D array
    a1 = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(my_sum(a1) - 10.0) < 1e-6

    # 2D array
    a2 = np.ones((4, 4))
    assert abs(my_sum(a2) - 16.0) < 1e-6

    # 3D array
    a3 = np.ones((2, 3, 4))
    assert abs(my_sum(a3) - 24.0) < 1e-6


def test_distances_and_weights():
    # Create 3 client states
    s1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    s2 = {"w": torch.tensor([1.1, 2.1, 3.1])}
    s3 = {"w": torch.tensor([5.0, 5.0, 5.0])}  # outlier

    states = [s1, s2, s3]
    num_examples = [100, 100, 100]

    # Test distance functions
    d_l1 = paper_l1_distance(states)
    d_gh = github_compat_distance(states)
    assert len(d_l1) == 3
    assert len(d_gh) == 3
    assert d_l1[2] > d_l1[0]  # Outlier has larger distance

    # Test base weights: s1 and s2 should have higher similarity weight than s3
    w, u, v, dists = base_weights(states, num_examples, distance_mode="paper_l1")
    assert abs(np.sum(w) - 1.0) < 1e-5
    assert u[0] > u[2]
    assert u[1] > u[2]


def test_regularize_weights():
    s1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    s2 = {"w": torch.tensor([10.0, 20.0, 30.0])}  # massive shift
    prev = {"w": torch.tensor([1.0, 2.0, 3.0])}

    w_base = np.array([0.5, 0.5])
    w_reg, deltas = regularize_weights(w_base, [s1, s2], prev)

    assert abs(np.sum(w_reg) - 1.0) < 1e-5
    # s1 had near-zero delta, so s1 weight should be regularized higher than s2
    assert w_reg[0] > w_reg[1]


def test_baselines():
    s1 = {"w": torch.tensor([1.0, 1.0])}
    s2 = {"w": torch.tensor([3.0, 3.0])}
    states = [s1, s2]
    num_examples = [10, 30]

    # FedAvg
    agg_fedavg = fedavg_aggregate(states, num_examples)
    expected = (1.0 * 10 + 3.0 * 30) / 40.0  # 2.5
    assert abs(agg_fedavg["w"][0] - expected) < 1e-5

    # Clipped Aggregation
    prev = {"w": torch.tensor([2.0, 2.0])}
    agg_clipped = clipped_aggregate(states, num_examples, previous_state=prev, percentile=50.0)
    assert "w" in agg_clipped

    # FedAvgM
    agg_m, vel = fedavgm_aggregate(states, num_examples, previous_state=prev, momentum=0.9, server_lr=1.0)
    assert "w" in agg_m
    assert "w" in vel
