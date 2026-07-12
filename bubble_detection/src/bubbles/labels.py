"""Label construction.

Labels are ALLOWED to look into the future (that is what a label is); the
walk-forward machinery purges any training rows whose label window overlaps
the test block, so no leakage reaches the models.

* ``fwd_ret``      : forward h-bar log return  y[t+h] - y[t]   (causal-forest outcome)
* ``dir_up``       : 1{fwd_ret > 0}                            (benchmark classifier target)
* ``bubble_ends_h``: among bars inside a detected explosive episode, 1 if the
                     episode terminates within the next h bars (PSY
                     date-stamping-based end-of-bubble label).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(df: pd.DataFrame, feats: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    lc = df["log_close"].to_numpy()
    n = lc.size
    lab = pd.DataFrame(index=df.index)

    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = lc[horizon:] - lc[: n - horizon]
    lab["fwd_ret"] = fwd
    lab["dir_up"] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))

    flag = feats["bubble_flag"].to_numpy()
    ends = np.full(n, np.nan)
    for t in range(n - horizon):
        if flag[t] == 1.0:
            ends[t] = float(np.any(flag[t + 1: t + 1 + horizon] == 0.0))
    lab["bubble_ends_h"] = ends
    return lab
