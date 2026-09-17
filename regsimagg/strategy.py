"""Flower strategy implementing RegSimAgg and FeTS Challenge configurations.

Integrates:
1. Dynamic per-round collaborator selection (sliding window without repetition).
2. Per-round hyperparameter configuration (learning rate, epochs/batches).
3. RegSimAgg aggregation (with both github_compat and paper_l1 distance modes).
4. Full challenge baselines: FedAvg, Clipped Aggregation, and FedAvgM.
5. FeTS multi-region segmentation metrics aggregation (WT, TC, ET Dice).
"""

from __future__ import annotations
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, Message
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from .collaborator_selector import CollaboratorSelector
from .hyperparameters import get_hyperparameters_for_round
from .regsimagg import (
    base_weights,
    regularize_weights,
    reg_sim_weighted_average_layerwise,
    aggregate_states,
    clipped_aggregate,
    fedavgm_aggregate,
    fedavg_aggregate,
)

logger = logging.getLogger(__name__)


class RegSimAgg(FedAvg):
    """RegSimAgg Strategy for Flower ServerApp."""

    def __init__(
        self,
        *,
        fraction_train: float = 0.2,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        regularization_round: int = 10,
        github_compat_distance: bool = True,
        layerwise_aggregation: bool = False,
        aggregation_method: str = "regsimagg",
        collaborator_selector_method: str = "without_repetition",
        hyperparameters_schedule: str = "constant",
        learning_rate: float = 5e-5,
        epochs_per_round: Optional[float] = 1.0,
        batches_per_round: Optional[int] = None,
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ):
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key="num-examples",
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
        )
        self.regularization_round = regularization_round
        self.github_compat_distance = github_compat_distance
        self.layerwise_aggregation = layerwise_aggregation
        self.aggregation_method = aggregation_method
        self.hyperparameters_schedule = hyperparameters_schedule
        self.base_lr = learning_rate
        self.base_epochs = epochs_per_round
        self.base_batches = batches_per_round

        # Collaborator selector instance
        self.selector = CollaboratorSelector(
            method=collaborator_selector_method,
            percentage=fraction_train,
        )

        self.previous_global: Optional[Dict[str, torch.Tensor]] = None
        self.momentum_velocity: Optional[Dict[str, np.ndarray]] = None
        self.last_weights: Optional[Dict[str, Any]] = None
        self.history_records: List[Dict[str, Any]] = []

    def get_round_train_config(self, server_round: int) -> ConfigRecord:
        """Compute training hyperparameters for the upcoming round."""
        lr, epochs, batches = get_hyperparameters_for_round(
            schedule=self.hyperparameters_schedule,
            fl_round=server_round,
            learning_rate=self.base_lr,
            epochs_per_round=self.base_epochs,
            batches_per_round=self.base_batches,
        )
        cfg: Dict[str, Any] = {"lr": float(lr)}
        if epochs is not None:
            cfg["epochs"] = float(epochs)
            cfg["batches"] = -1
        else:
            cfg["epochs"] = -1.0
            cfg["batches"] = int(batches) if batches is not None else 16
        return ConfigRecord(cfg)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies_list = list(replies)
        valid, _ = self._check_and_log_replies(replies_list, is_train=True)
        if not valid:
            return None, None

        arrays_key = self.arrayrecord_key
        records = [msg.content[arrays_key] for msg in valid]
        states = [r.to_torch_state_dict() for r in records]
        num_examples = [float(msg.content["metrics"].get("num-examples", 1.0)) for msg in valid]

        dist_mode = "github_compat" if self.github_compat_distance else "paper_l1"

        if self.aggregation_method == "fedavg":
            agg_state = fedavg_aggregate(states, num_examples)
            weights = (np.array(num_examples) / sum(num_examples)).tolist()
            deltas = [0.0] * len(states)
            sim_w, sample_w, distances = weights, weights, [0.0] * len(states)

        elif self.aggregation_method == "clipped":
            agg_state = clipped_aggregate(
                states, num_examples, previous_state=self.previous_global, percentile=80.0
            )
            weights = (np.array(num_examples) / sum(num_examples)).tolist()
            deltas = [0.0] * len(states)
            sim_w, sample_w, distances = weights, weights, [0.0] * len(states)

        elif self.aggregation_method == "fedavgm":
            agg_state, self.momentum_velocity = fedavgm_aggregate(
                states,
                num_examples,
                previous_state=self.previous_global,
                velocity=self.momentum_velocity,
                momentum=0.9,
                server_lr=1.0,
            )
            weights = (np.array(num_examples) / sum(num_examples)).tolist()
            deltas = [0.0] * len(states)
            sim_w, sample_w, distances = weights, weights, [0.0] * len(states)

        else:  # Default: "regsimagg"
            if self.layerwise_aggregation:
                agg_state = reg_sim_weighted_average_layerwise(
                    states=states,
                    num_examples=num_examples,
                    fl_round=server_round,
                    previous_state=self.previous_global,
                    distance_mode=dist_mode,
                    regularization_round=self.regularization_round,
                )
                weights, sim_w, sample_w, distances = base_weights(
                    states, num_examples, distance_mode=dist_mode
                )
                deltas = [0.0] * len(states)
            else:
                w_base, sim_w_arr, sample_w_arr, distances_arr = base_weights(
                    states, num_examples, distance_mode=dist_mode
                )
                if self.previous_global is not None and server_round > self.regularization_round:
                    weights_arr, deltas_arr = regularize_weights(
                        w_base, states, self.previous_global
                    )
                else:
                    weights_arr = w_base
                    deltas_arr = np.zeros(len(states))

                agg_state = aggregate_states(states, weights_arr)
                weights = weights_arr.tolist()
                sim_w = sim_w_arr.tolist()
                sample_w = sample_w_arr.tolist()
                distances = distances_arr.tolist()
                deltas = deltas_arr.tolist()

        # Save metadata record for reproduction audit
        self.last_weights = {
            "round": server_round,
            "weights": weights if isinstance(weights, list) else weights.tolist(),
            "similarity_weights": sim_w if isinstance(sim_w, list) else sim_w.tolist(),
            "sample_weights": sample_w if isinstance(sample_w, list) else sample_w.tolist(),
            "distances": distances if isinstance(distances, list) else distances.tolist(),
            "regularization_deltas": deltas if isinstance(deltas, list) else deltas.tolist(),
        }
        self.history_records.append(self.last_weights)

        # Convert numpy state back to torch/ArrayRecord
        aggregated = ArrayRecord({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in agg_state.items()})
        self.previous_global = aggregated.to_torch_state_dict()

        metrics = aggregate_metricrecords(
            [m.content for m in valid], "num-examples"
        )
        return aggregated, metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> Tuple[Optional[MetricRecord], Optional[Dict[str, Any]]]:
        """Aggregate multi-region validation metrics (WT, TC, ET Dice)."""
        replies_list = list(replies)
        valid, _ = self._check_and_log_replies(replies_list, is_train=False)
        if not valid:
            return None, None

        metrics = aggregate_metricrecords([m.content for m in valid], "num-examples")
        return metrics, None

    def summary(self) -> None:
        super().summary()
        print(
            f"RegSimAgg Strategy Summary:\n"
            f"  Aggregation Method: {self.aggregation_method}\n"
            f"  Collaborator Selector: {self.selector.method} (Fraction: {self.fraction_train})\n"
            f"  Distance Mode: {'github_compat' if self.github_compat_distance else 'paper_l1'}\n"
            f"  Regularization Round: {self.regularization_round}\n"
            f"  Layerwise Aggregation: {self.layerwise_aggregation}\n"
            f"  Hyperparameter Schedule: {self.hyperparameters_schedule}"
        )
