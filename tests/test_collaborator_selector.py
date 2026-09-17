"""Unit tests for collaborator selection algorithms."""

import pytest
import numpy as np
from regsimagg.collaborator_selector import (
    equal_partitions,
    custom_percentage_collaborator_selector_without_repetition,
    custom_percentage_collaborator_selector,
    all_collaborators_train,
    one_collaborator_on_odd_rounds,
    CollaboratorSelector,
)


def test_equal_partitions():
    # Length 5 with windowsize 2 -> should pad to 6
    colabs = ["c0", "c1", "c2", "c3", "c4"]
    padded = equal_partitions(colabs, windowsize=2)
    assert len(padded) == 6
    assert padded[-1] == "c0"

    # Length 6 with windowsize 2 -> already divisible, no padding
    colabs_even = ["c0", "c1", "c2", "c3", "c4", "c5"]
    padded_even = equal_partitions(colabs_even, windowsize=2)
    assert len(padded_even) == 6


def test_without_repetition_unique_subsets():
    colabs = [f"client_{i}" for i in range(10)]
    selector = CollaboratorSelector(method="without_repetition", percentage=0.2, seed=42)

    seen_subsets = []
    for r in range(5):
        selected = selector.select(colabs, fl_round=r)
        assert len(selected) == 2  # 20% of 10 is 2
        sorted_sub = tuple(sorted(selected))
        assert sorted_sub not in seen_subsets, f"Repeated subset in round {r}: {sorted_sub}"
        seen_subsets.append(sorted_sub)


def test_all_collaborators_train():
    colabs = ["c1", "c2", "c3"]
    selected = all_collaborators_train(colabs, fl_round=0)
    assert selected == colabs


def test_one_collaborator_on_odd_rounds():
    colabs = ["c1", "c2", "c3"]
    times = {0: {"c1": 15.0, "c2": 8.0, "c3": 20.0}}

    # Even round: all train
    assert one_collaborator_on_odd_rounds(colabs, fl_round=0) == colabs

    # Odd round: fastest (c2 with 8.0s) trains
    selected = one_collaborator_on_odd_rounds(
        colabs, fl_round=1, collaborator_times_per_round=times
    )
    assert selected == ["c2"]
