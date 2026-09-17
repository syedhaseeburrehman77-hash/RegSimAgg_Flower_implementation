"""Collaborator selection algorithms for federated learning.

Implements the client selection methods from the FeTS 2022 Challenge
and the RegSimAgg paper (Khan et al., 2022):
- Sliding-window 20% selection without repeating combinations
- Equal partitioning
- All-collaborators training
- Alternating collaborator selection based on execution time
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

logger = logging.getLogger(__name__)

all_time_colab_list: List[List[str]] = []


def equal_partitions(collaborators: Sequence[str] | np.ndarray, windowsize: float | int) -> np.ndarray:
    """Pad the collaborator list if its length is not a multiple of window size.

    Matches the FeTS 2022 implementation.
    """
    colabs = np.asarray(collaborators)
    w = int(windowsize)
    if w <= 0:
        return colabs
    remainder = len(colabs) % w
    if remainder != 0:
        padding = colabs[0 : (w - remainder)]
        colabs = np.append(colabs, padding)
    return colabs


def custom_percentage_collaborator_selector_without_repetition(
    collaborators: Sequence[str],
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    percentage: float = 0.2,
    seed: Optional[int] = None,
) -> List[str]:
    """Select a fraction (default 20%) of collaborators per round using a sliding window.

    Ensures that every collaborator participates an equal number of times across a cycle.
    Once a cycle finishes, the order is randomly shuffled.
    No selected subset of collaborators is ever repeated in future rounds.
    """
    global all_time_colab_list

    if seed is not None and fl_round == 0:
        np.random.seed(seed)

    original_array = list(collaborators)
    total_clients = len(original_array)
    if total_clients == 0:
        return []

    windowsize = max(1, int(total_clients * percentage))
    padded_colabs = equal_partitions(original_array, windowsize)
    pieces = int(np.ceil(len(padded_colabs) / windowsize))

    # Determine base permutation for this round
    cycle = fl_round // pieces if pieces > 0 else 0
    cycle_step = fl_round % pieces if pieces > 0 else 0

    rng = np.random.default_rng(seed + cycle if seed is not None else None)

    # Search for a non-repeated subset
    max_attempts = 100
    for attempt in range(max_attempts):
        shuffled = np.array(original_array)
        if cycle > 0 or attempt > 0:
            rng.shuffle(shuffled)
        padded = equal_partitions(shuffled, windowsize)

        start_idx = (cycle_step * windowsize) % len(padded)
        selected = padded[start_idx : start_idx + windowsize].tolist()

        # Check if this exact combination was selected before
        sorted_selected = sorted(selected)
        if not any(sorted(c) == sorted_selected for c in all_time_colab_list):
            all_time_colab_list.append(selected)
            logger.info("Round %d selected collaborators: %s", fl_round, selected)
            return selected

    # Fallback if no unique subset found after max_attempts
    logger.warning("Could not find an unrepeated collaborator set, using default slice.")
    selected = padded[start_idx : start_idx + windowsize].tolist()
    all_time_colab_list.append(selected)
    return selected


def custom_percentage_collaborator_selector(
    collaborators: Sequence[str],
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    percentage: float = 0.2,
) -> List[str]:
    """Sliding-window percentage selector without non-repetition constraint."""
    original = np.array(collaborators)
    windowsize = max(1, int(len(original) * percentage))
    padded = equal_partitions(original, windowsize)
    pieces = int(np.ceil(len(padded) / windowsize))

    if fl_round >= pieces and (fl_round % pieces == 0):
        np.random.shuffle(padded)

    start_idx = (fl_round * windowsize) % len(padded)
    return padded[start_idx : start_idx + windowsize].tolist()


def all_collaborators_train(
    collaborators: Sequence[str],
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[str]:
    """Baseline: all collaborators train every round."""
    return list(collaborators)


def one_collaborator_on_odd_rounds(
    collaborators: Sequence[str],
    db_iterator: Any = None,
    fl_round: int = 0,
    collaborators_chosen_each_round: Optional[Dict[int, List[str]]] = None,
    collaborator_times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[str]:
    """Select fastest collaborator on odd rounds, everyone on even rounds."""
    if fl_round % 2 == 1 and collaborator_times_per_round and (fl_round - 1) in collaborator_times_per_round:
        fastest_time = float("inf")
        fastest_col = None
        for col, t in collaborator_times_per_round[fl_round - 1].items():
            if t < fastest_time:
                fastest_time = t
                fastest_col = col
        if fastest_col is not None:
            return [fastest_col]
    return list(collaborators)


class CollaboratorSelector:
    """Stateful selector class designed for clean integration with Flower ServerApp."""

    def __init__(
        self,
        method: str = "without_repetition",
        percentage: float = 0.2,
        seed: Optional[int] = 42,
    ):
        self.method = method
        self.percentage = percentage
        self.seed = seed
        self.history: List[List[str]] = []
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        all_collaborators: Sequence[str],
        fl_round: int,
        times_per_round: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[str]:
        if self.method == "all":
            selected = all_collaborators_train(all_collaborators, fl_round=fl_round)
        elif self.method == "odd_rounds":
            selected = one_collaborator_on_odd_rounds(
                all_collaborators,
                fl_round=fl_round,
                collaborator_times_per_round=times_per_round,
            )
        elif self.method == "percentage":
            selected = custom_percentage_collaborator_selector(
                all_collaborators, fl_round=fl_round, percentage=self.percentage
            )
        else:  # default: "without_repetition"
            selected = custom_percentage_collaborator_selector_without_repetition(
                all_collaborators,
                fl_round=fl_round,
                percentage=self.percentage,
                seed=self.seed,
            )

        self.history.append(selected)
        return selected

    def reset(self) -> None:
        global all_time_colab_list
        all_time_colab_list.clear()
        self.history.clear()
        self.rng = np.random.default_rng(self.seed)
