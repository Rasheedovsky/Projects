"""Bai-Perron multiple structural break detection as rolling features.

Bai & Perron (1998, 2003): simultaneous estimation of multiple breaks in a
linear regression by *global* minimisation of the sum of squared residuals
over all admissible break partitions, solved with dynamic programming; the
number of breaks is selected by an information criterion (BIC here).

We run the procedure on the AR(1) levels model over a trailing window
ending at each bar (candidate breaks on a grid, minimum segment length
enforced), which is exactly the Bai-Perron DP restricted to a grid for
speed.  The DP itself is vectorised *across windows*: the value tables of
all windows advance together in numpy.

Features per bar:
  bp_nbreaks : BIC-selected number of breaks in the trailing window (0..max)
  bp_since   : bars since the most recent detected break (window length if none)
  bp_dmean   : shift in mean log-return between the last two regimes (0 if < 1 break)
"""
from __future__ import annotations

import numpy as np

from .prefix_ols import make_ar1_engine

K_AR1 = 2


def rolling_bai_perron(y: np.ndarray, window: int = 336, max_breaks: int = 3,
                       min_seg: int = 30, grid_step: int = 5, stride: int = 8):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    eng = make_ar1_engine(y)
    m_obs = window - 1                                    # obs per window

    # Candidate boundary offsets (break after relative obs o), plus sentinels.
    grid = np.arange(min_seg - 1, m_obs - min_seg, grid_step, dtype=np.int64)
    B = np.concatenate([[-1], grid, [m_obs - 1]])         # nodes
    nb = B.size
    START, END = 0, nb - 1

    all_ends = np.arange(window - 1, n, stride, dtype=np.int64)

    # node-pair structure is identical for every window (fixed offsets)
    ii, jj = np.meshgrid(np.arange(nb), np.arange(nb), indexing="ij")
    seg_len = B[jj] - B[ii]
    valid = (jj > ii) & (seg_len >= min_seg)
    vi, vj = ii[valid], jj[valid]
    n_pairs = vi.size

    dy_mean = lambda a, b: (y[b] - y[a - 1]) / max(b - a + 1, 1)   # mean log-return
    nb_out = np.full(n, np.nan)
    since_out = np.full(n, np.nan)
    dmean_out = np.full(n, np.nan)
    n0 = float(m_obs)
    p_m = np.array([(m + 1) * K_AR1 + m for m in range(max_breaks + 1)], dtype=float)

    block = max(1, int(4e6) // max(nb * nb, 1))            # bound cost-tensor memory
    for b0 in range(0, all_ends.size, block):
        ends = all_ends[b0: b0 + block]
        n_win = ends.size
        s_obs = ends - window + 2                          # first regression obs

        # --- batched segment costs C[w, i, j] = SSR(obs[B_i+1 .. B_j]) -----
        Sg = (s_obs[:, None] + (B[vi] + 1)[None, :]).ravel()
        Eg = (s_obs[:, None] + B[vj][None, :]).ravel()
        ssr = eng.ssr(Sg, Eg).reshape(n_win, n_pairs)
        C = np.full((n_win, nb, nb), np.inf)
        C[:, vi, vj] = np.where(np.isfinite(ssr), ssr, np.inf)

        # --- DP across the block's windows simultaneously ------------------
        F = [C[:, START, :]]                               # r = 0
        Args = [np.full((n_win, nb), -1, dtype=np.int64)]
        for r in range(1, max_breaks + 1):
            tot = F[r - 1][:, :, None] + C                 # (w, i, j)
            Args.append(np.argmin(tot, axis=1))
            F.append(np.min(tot, axis=1))

        ssr_m = np.stack([F[r][:, END] for r in range(max_breaks + 1)], axis=1)
        ssr_m = np.maximum(ssr_m, 1e-14)
        bic = n0 * np.log(ssr_m / n0) + p_m[None, :] * np.log(n0)
        bic = np.where(np.isfinite(bic), bic, np.inf)
        m_star = np.argmin(bic, axis=1)

        for w, e in enumerate(ends):
            m = int(m_star[w])
            nb_out[e] = m
            if m == 0:
                since_out[e] = float(m_obs)
                dmean_out[e] = 0.0
                continue
            node = int(Args[m][w, END])                    # last break node
            lb_abs = s_obs[w] + B[node]                    # absolute obs index of break
            since_out[e] = float(e - lb_abs)
            seg2 = dy_mean(lb_abs + 1, e)                  # regime after last break
            prev_off = B[int(Args[m - 1][w, node])] + 1 if m >= 2 else 0
            seg1 = dy_mean(s_obs[w] + prev_off, lb_abs)    # regime before last break
            dmean_out[e] = seg2 - seg1

    # forward-fill stride gaps (causal: values come from an earlier window end)
    for arr in (nb_out, since_out, dmean_out):
        idx = np.where(~np.isnan(arr))[0]
        if idx.size:
            filled = np.full(n, np.nan)
            pos = np.searchsorted(idx, np.arange(n), side="right") - 1
            ok = pos >= 0
            filled[ok] = arr[idx[pos[ok]]]
            arr[:] = np.where(np.arange(n) >= idx[0], filled, np.nan)
    return nb_out, since_out, dmean_out
