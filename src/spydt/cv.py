"""Purged combinatorial cross-validation (AFML ch. 7 & 12).

Days carry non-overlapping intraday labels, so purging is thin; the embargo
is tied to the longest EWMA half-life used in preprocessing (config), applied
AFTER each test block (information from test days decays forward into
subsequently-fit estimators).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Split:
    split_id: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_groups(n_days: int, n_groups: int) -> np.ndarray:
    """Contiguous group label per day index."""
    return np.minimum((np.arange(n_days) * n_groups) // n_days, n_groups - 1)


def cpcv_splits(
    n_days: int,
    *,
    n_groups: int,
    k_test: int,
    purge_days: int,
    embargo_days: int,
) -> list[Split]:
    """All C(n_groups, k_test) splits with purge+embargo around test blocks."""
    groups = make_groups(n_days, n_groups)
    idx = np.arange(n_days)
    out: list[Split] = []
    for sid, combo in enumerate(combinations(range(n_groups), k_test)):
        test_mask = np.isin(groups, combo)
        banned = test_mask.copy()
        for g in combo:
            g_idx = idx[groups == g]
            lo, hi = g_idx.min(), g_idx.max()
            banned[max(0, lo - purge_days) : lo] = True          # purge before
            banned[hi + 1 : hi + 1 + purge_days + embargo_days] = True  # purge+embargo after
        train_mask = ~banned & ~test_mask
        out.append(
            Split(sid, tuple(combo), idx[train_mask].copy(), idx[test_mask].copy())
        )
    return out


def paths_from_splits(n_groups: int, k_test: int) -> list[dict[int, int]]:
    """CPCV backtest paths: path -> {group: split_id supplying its forecast}.

    Each group appears in C(n-1, k-1) splits as test; those appearances are
    dealt round-robin into that many complete paths.
    """
    combos = list(combinations(range(n_groups), k_test))
    appearances: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for sid, combo in enumerate(combos):
        for g in combo:
            appearances[g].append(sid)
    n_paths = len(appearances[0])
    return [{g: appearances[g][p] for g in range(n_groups)} for p in range(n_paths)]


def assert_no_leakage(splits: list[Split], purge_days: int,
                      embargo_days: int = 0) -> None:
    """Canary: no train index inside the purge window before/around a test
    block, nor inside the purge+embargo window after it.

    Raises AssertionError on violation — used by the train-on-future canary
    test, which feeds a deliberately corrupted split and expects a throw.
    """
    after = purge_days + embargo_days
    for s in splits:
        if len(np.intersect1d(s.train_idx, s.test_idx)):
            raise AssertionError(f"split {s.split_id}: train/test overlap")
        t = np.sort(s.test_idx)
        blocks = np.split(t, np.where(np.diff(t) > 1)[0] + 1)
        for b in blocks:
            near = s.train_idx[
                (s.train_idx >= b.min() - purge_days) & (s.train_idx <= b.max() + after)
            ]
            if len(near):
                raise AssertionError(
                    f"split {s.split_id}: train days {near[:5]} inside the "
                    f"purge/embargo window of test block [{b.min()}, {b.max()}]"
                )
