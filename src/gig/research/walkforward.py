"""Purged expanding-window walk-forward with embargo (Lopez de Prado)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


class WalkForwardSplit:
    """
    Expanding train window, contiguous test blocks, embargo gap between them.

    Embargo drops `embargo` sessions after train_end so overlapping labels
    (e.g. 5-day forward returns) cannot leak into the test set.
    """

    def __init__(
        self,
        min_train: int = 252,
        test_size: int = 63,
        embargo: int = 5,
        step: int | None = None,
    ) -> None:
        self.min_train = min_train
        self.test_size = test_size
        self.embargo = embargo
        self.step = step or test_size

    def split(self, index: pd.DatetimeIndex) -> Iterator[Fold]:
        idx = pd.DatetimeIndex(index).sort_values()
        n = len(idx)
        start = self.min_train
        while start + self.embargo + self.test_size <= n:
            train_end_i = start - 1
            test_start_i = start + self.embargo
            test_end_i = test_start_i + self.test_size - 1
            if test_end_i >= n:
                break
            yield Fold(
                train_end=idx[train_end_i],
                test_start=idx[test_start_i],
                test_end=idx[test_end_i],
                train_index=idx[: start],
                test_index=idx[test_start_i : test_end_i + 1],
            )
            start += self.step
