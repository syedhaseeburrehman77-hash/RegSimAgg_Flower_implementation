"""RegSimAgg and baseline aggregation algorithms.

Implements:
1. RegSimAgg (Regularized Similarity Weighted Aggregation) from Khan et al. (2022)
   and the FeTS 2022 Challenge GitHub repository (dskhanirfan/FeTS2022).
2. Exact my_sum recursive reduction matching the public code.
3. Both github_compat distance mode and paper_l1 distance mode.
4. Base weights calculation combining similarity weights (Eq. 3) and sample weights (Eq. 4).
5. Round > 10 regularization (Eq. 6).
6. Baseline aggregators from the FeTS competition:
   - FedAvg (weighted average)
   - Clipped Aggregation (80th percentile clipping)
   - FedAvgM (server momentum = 0.9, server lr = 1.0)
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

EPS = 1e-5


def my_sum(input_array: np.ndarray | float | int) -> float:
    """Recursive multi-dimensional summation matching the FeTS GitHub implementation.

    In the author's public code, this function reduces any multi-dimensional tensor
    down to a single scalar sum by recursively summing across dimensions.
    """
    arr = np.asarray(input_array)
    if arr.ndim > 1:
        means = np.sum(arr, axis=1)
        result = np.sum(means, axis=0)
        if result.ndim > 1:
            result = my_sum(result)
    else:
        result = arr.copy()

    if np.isscalar(result) or result.ndim == 0:
        return float(result)
    else:
        return float(np.sum(result))


def flatten_state(state: Dict[str, Union[torch.Tensor, np.ndarray]]) -> np.ndarray:
    """Flatten all model parameters into a single 1D numpy array."""
    chunks = []
    for v in state.values():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            arr = np.asarray(v, dtype=np.float64).ravel()
        chunks.append(arr)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def state_to_numpy(
    state: Dict[str, Union[torch.Tensor, np.ndarray]]
) -> OrderedDict[str, np.ndarray]:
    """Convert a PyTorch state dict or dictionary of arrays to numpy arrays."""
    out = OrderedDict()
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


# ---------------------------------------------------------------------------
# Distance & Weight Calculations
# ---------------------------------------------------------------------------

def paper_l1_distance(states: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Distance from each client model to the unweighted mean model.

    Paper Eq. (2):
        d_c = ||p_c - p_hat||_1 = sum |p_c - p_hat|
    """
    vecs = [flatten_state(s) for s in states]
    mean_vec = np.mean(np.stack(vecs), axis=0)
    return np.asarray([np.sum(np.abs(v - mean_vec)) for v in vecs], dtype=np.float64)


def github_compat_distance(states: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Distance matching the author's public GitHub implementation:

        d_c = | my_sum(mean_tensor) - my_sum(client_tensor) |
    """
    vecs = [flatten_state(s) for s in states]
    mean_vec = np.mean(np.stack(vecs), axis=0)
    return np.asarray([abs(my_sum(mean_vec) - my_sum(v)) for v in vecs], dtype=np.float64)


def similarity_weights(
    states: Sequence[Dict[str, Any]],
    distance_mode: str = "github_compat",
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized similarity weights u_c from Eq. (2) and Eq. (3).

    Returns:
        (u, distances) where u sums to 1.
    """
    if distance_mode == "paper_l1":
        distances = paper_l1_distance(states)
    else:
        distances = github_compat_distance(states)

    total_dist = float(np.sum(distances))
    # s_c = total_dist / (dist_c + eps) (Eq. 2)
    s = total_dist / (distances + eps)
    # u_c = s_c / (sum(s) + eps) (Eq. 3)
    u = s / (np.sum(s) + eps)
    return u, distances


def sample_weights(num_examples: Sequence[float | int], eps: float = EPS) -> np.ndarray:
    """Compute sample size weights v_c from Eq. (4):

        v_c = N_c / sum(N_i)
    """
    n = np.asarray(num_examples, dtype=np.float64)
    total = np.sum(n)
    if total <= 0:
        return np.ones_like(n) / len(n)
    return n / total


def base_weights(
    states: Sequence[Dict[str, Any]],
    num_examples: Sequence[float | int],
    distance_mode: str = "github_compat",
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute base RegSimAgg aggregation weights from Eq. (5):

        w_c = (u_c + v_c) / sum(u_i + v_i)

    Returns:
        (w, u, v, distances)
    """
    u, distances = similarity_weights(states, distance_mode=distance_mode, eps=eps)
    v = sample_weights(num_examples, eps=eps)
    w = u + v
    w = w / (np.sum(w) + eps)
    return w, u, v, distances


def regularize_weights(
    w: np.ndarray,
    states: Sequence[Dict[str, Any]],
    previous_state: Dict[str, Any],
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply post-round-10 delta regularization from Eq. (6).

    Paper Eq. (6):
        w_c_reg = w_c / (||p_c - p_prev|| + eps), then normalized.

    This dampens updates from clients undergoing excessive parameter shifts,
    stabilizing convergence in later federation rounds.
    """
    prev_vec = flatten_state(previous_state)
    deltas = np.asarray([
        np.mean(np.abs(flatten_state(s) - prev_vec)) for s in states
    ], dtype=np.float64)

    reg = w / (deltas + eps)
    reg_sum = np.sum(reg)
    if reg_sum > 0 and np.isfinite(reg_sum):
        reg = reg / (reg_sum + eps)
    else:
        logger.warning("Regularization sum non-finite, falling back to base weights.")
        reg = w.copy()
    return reg, deltas


# ---------------------------------------------------------------------------
# Layer-wise RegSimAgg (Exact OpenFL Callback Reproduction)
# ---------------------------------------------------------------------------

def reg_sim_weighted_average_layerwise(
    states: Sequence[Dict[str, Any]],
    num_examples: Sequence[float | int],
    fl_round: int,
    previous_state: Optional[Dict[str, Any]] = None,
    distance_mode: str = "github_compat",
    regularization_round: int = 10,
    eps: float = EPS,
) -> OrderedDict[str, np.ndarray]:
    """Performs RegSimAgg layer-by-layer, faithfully reproducing OpenFL's callback mechanism.

    In OpenFL, the aggregation function receives `local_tensors` for each layer independently.
    """
    keys = list(states[0].keys())
    np_states = [state_to_numpy(s) for s in states]
    weight_values = sample_weights(num_examples, eps=eps)

    prev_np = state_to_numpy(previous_state) if previous_state is not None else None
    out = OrderedDict()

    for k in keys:
        tensor_values = [s[k] for s in np_states]
        total_avg = np.average(tensor_values, axis=0)

        # Distance calculation
        distances = []
        for tv in tensor_values:
            if distance_mode == "github_compat":
                d = abs(my_sum(total_avg) - my_sum(tv))
            else:
                d = float(np.sum(np.abs(total_avg - tv)))
            distances.append(d)

        distances = np.asarray(distances, dtype=np.float64)
        total_dist = float(np.sum(distances))

        # Similarity weights (Eq. 2 & 3)
        sim_weights = total_dist / (distances + eps)
        sim_sum = float(np.sum(sim_weights))

        # Base weights (Eq. 5)
        weights_norm = []
        for i in range(len(tensor_values)):
            weights_norm.append(weight_values[i] + sim_weights[i] / (sim_sum + eps))
        weights_norm = np.asarray(weights_norm, dtype=np.float64)
        weights_norm_1 = weights_norm / (np.sum(weights_norm) + eps)

        # Regularization when round > 10 (Eq. 6)
        if fl_round > regularization_round and prev_np is not None and k in prev_np:
            prev_val = prev_np[k]
            deltas = [prev_val - tv for tv in tensor_values]
            avg_deltas = np.average(deltas, weights=weight_values, axis=0)

            weights_norm_2 = []
            for i in range(len(weights_norm_1)):
                weights_norm_2.append(weights_norm_1[i] / (avg_deltas + eps))

            try:
                # Attempt tensor-weighted average matching author's GitHub code
                layer_agg = np.round(np.average(tensor_values, weights=weights_norm_2, axis=0), 4)
            except Exception:
                # Graceful fallback to base weights or FedAvg
                try:
                    layer_agg = np.round(np.average(tensor_values, weights=weights_norm_1, axis=0), 4)
                except Exception:
                    layer_agg = np.round(np.average(tensor_values, weights=weight_values, axis=0), 4)
        else:
            try:
                layer_agg = np.round(np.average(tensor_values, weights=weights_norm_1, axis=0), 4)
            except Exception:
                layer_agg = np.round(np.average(tensor_values, weights=weight_values, axis=0), 4)

        out[k] = layer_agg.astype(tensor_values[0].dtype)

    return out


# ---------------------------------------------------------------------------
# General State Aggregation
# ---------------------------------------------------------------------------

def aggregate_states(
    states: Sequence[Dict[str, Any]],
    weights: Sequence[float] | np.ndarray,
) -> OrderedDict[str, np.ndarray]:
    """Compute weighted average of state dictionaries."""
    keys = list(states[0].keys())
    np_states = [state_to_numpy(s) for s in states]
    weights_arr = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(weights_arr)
    if w_sum > 0:
        weights_arr = weights_arr / w_sum

    out = OrderedDict()
    for k in keys:
        arrs = [s[k] for s in np_states]
        stacked = np.stack(arrs, axis=0)
        out[k] = np.tensordot(weights_arr, stacked, axes=(0, 0)).astype(arrs[0].dtype)
    return out


# ---------------------------------------------------------------------------
# Baseline Aggregators (for Challenge Reproduction & Comparison)
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    states: Sequence[Dict[str, Any]],
    num_examples: Sequence[float | int],
) -> OrderedDict[str, np.ndarray]:
    """Standard FedAvg (weighted average by sample counts)."""
    weights = sample_weights(num_examples)
    return aggregate_states(states, weights)


def clipped_aggregate(
    states: Sequence[Dict[str, Any]],
    num_examples: Sequence[float | int],
    previous_state: Optional[Dict[str, Any]],
    percentile: float = 80.0,
) -> OrderedDict[str, np.ndarray]:
    """Clipped Aggregation from FeTS reference:

    Clips local parameter updates to the N-th percentile of absolute deltas.
    """
    if previous_state is None:
        return fedavg_aggregate(states, num_examples)

    prev_np = state_to_numpy(previous_state)
    np_states = [state_to_numpy(s) for s in states]
    weights = sample_weights(num_examples)
    keys = list(states[0].keys())

    out = OrderedDict()
    for k in keys:
        prev_val = prev_np[k]
        local_tensors = [s[k] for s in np_states]
        deltas = [tv - prev_val for tv in local_tensors]

        clip_value = float(np.percentile(np.abs(deltas), percentile))
        clipped_tensors = []
        for delta in deltas:
            clipped = prev_val + np.clip(delta, -clip_value, clip_value)
            clipped_tensors.append(clipped)

        out[k] = np.average(clipped_tensors, weights=weights, axis=0).astype(prev_val.dtype)
    return out


def fedavgm_aggregate(
    states: Sequence[Dict[str, Any]],
    num_examples: Sequence[float | int],
    previous_state: Optional[Dict[str, Any]],
    velocity: Optional[Dict[str, np.ndarray]] = None,
    momentum: float = 0.9,
    server_lr: float = 1.0,
) -> Tuple[OrderedDict[str, np.ndarray], Dict[str, np.ndarray]]:
    """FedAvg with Server Momentum (FedAvgM) from FeTS 2021/2022.

    V_{t+1} = momentum * V_t + Average_Delta_t
    W_{t+1} = W_t - server_lr * V_{t+1}
    """
    if previous_state is None:
        agg = fedavg_aggregate(states, num_examples)
        vel = {k: np.zeros_like(v) for k, v in agg.items()}
        return agg, vel

    prev_np = state_to_numpy(previous_state)
    np_states = [state_to_numpy(s) for s in states]
    weights = sample_weights(num_examples)
    keys = list(states[0].keys())

    new_velocity = {}
    out = OrderedDict()

    for k in keys:
        prev_val = prev_np[k]
        local_tensors = [s[k] for s in np_states]
        deltas = [prev_val - tv for tv in local_tensors]
        avg_delta = np.average(deltas, weights=weights, axis=0)

        v_prev = velocity[k] if velocity is not None and k in velocity else np.zeros_like(prev_val)
        v_new = momentum * v_prev + avg_delta
        new_velocity[k] = v_new

        w_new = prev_val - server_lr * v_new
        out[k] = w_new.astype(prev_val.dtype)

    return out, new_velocity
