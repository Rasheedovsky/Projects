"""Purged expanding-window walk-forward validation.

Folds are contiguous, strictly ordered in time.  Because labels look
``horizon`` bars ahead, the last ``horizon`` bars of every training slice
have label windows overlapping the test block; those rows are *purged*
(dropped from training) so no future information reaches the model
(cf. Lopez de Prado, Advances in Financial Machine Learning, ch. 7).
"""
from __future__ import annotations

import numpy as np


def purged_walkforward(usable_idx: np.ndarray, train_min: int, test_size: int,
                       purge: int):
    """Yield (train_idx, test_idx) pairs of absolute indices.

    usable_idx : sorted absolute indices of rows with complete features+labels
    train_min  : minimum number of training rows before the first test block
    test_size  : rows per test block
    purge      : number of training rows dropped immediately before each test
                 block (label horizon)
    """
    usable_idx = np.asarray(usable_idx)
    n = usable_idx.size
    folds = []
    start = train_min
    while start + 1 <= n - 1:
        test = usable_idx[start: start + test_size]
        if test.size < max(20, test_size // 3):
            break
        train = usable_idx[: max(0, start - purge)]
        if train.size >= 100:
            folds.append((train, test))
        start += test_size
    return folds
