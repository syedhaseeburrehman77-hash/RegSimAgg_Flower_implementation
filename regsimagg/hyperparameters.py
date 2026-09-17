"""Hyperparameter schedules for federated learning rounds.

Implements the training hyperparameter methods from the FeTS 2022 Challenge
and the RegSimAgg paper (Khan et al., 2022):
- constant_hyper_parameters: default lr=5e-5, epochs=1.0
- train_less_each_round: decaying epochs for the first 10 rounds
- fixed_number_of_batches: fixed batch count (Irrespective of client dataset size)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

HyperparamTuple = Tuple[float, Optional[float], Optional[int]]


def constant_hyper_parameters(
    collaborators: Optional[Sequence[str]] = None,
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    learning_rate: float = 5e-5,
    epochs_per_round: float = 1.0,
) -> HyperparamTuple:
    """Standard FeTS 2022 hyperparameters.

    Returns:
        (learning_rate, epochs_per_round, batches_per_round)
        Note: exactly one of epochs_per_round and batches_per_round is None.
    """
    return (learning_rate, float(epochs_per_round), None)


def train_less_each_round(
    collaborators: Optional[Sequence[str]] = None,
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    learning_rate: float = 5e-5,
    initial_epochs: float = 1.0,
    decay_rate: float = 0.9,
    max_decay_rounds: int = 10,
) -> HyperparamTuple:
    """Decaying epochs per round for initial rounds."""
    decay_power = min(fl_round, max_decay_rounds)
    epochs = initial_epochs * (decay_rate ** decay_power)
    return (learning_rate, epochs, None)


def fixed_number_of_batches(
    collaborators: Optional[Sequence[str]] = None,
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    learning_rate: float = 5e-5,
    batches_per_round: int = 16,
) -> HyperparamTuple:
    """Fixed number of batches per client round, irrespective of local dataset size."""
    return (learning_rate, None, int(batches_per_round))


def get_hyperparameters_for_round(
    schedule: str = "constant",
    fl_round: int = 0,
    learning_rate: float = 5e-5,
    epochs_per_round: Optional[float] = 1.0,
    batches_per_round: Optional[int] = None,
) -> HyperparamTuple:
    """Convenience dispatcher to produce hyperparameters for a Flower round."""
    if schedule == "decay":
        return train_less_each_round(fl_round=fl_round, learning_rate=learning_rate)
    elif schedule == "fixed_batches":
        batches = batches_per_round if batches_per_round is not None else 16
        return fixed_number_of_batches(fl_round=fl_round, learning_rate=learning_rate, batches_per_round=batches)
    else:  # "constant"
        ep = epochs_per_round if epochs_per_round is not None else 1.0
        return constant_hyper_parameters(fl_round=fl_round, learning_rate=learning_rate, epochs_per_round=ep)
