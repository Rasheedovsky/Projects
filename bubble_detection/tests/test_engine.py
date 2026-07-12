"""Correctness tests: prefix-sum engine vs explicit statsmodels regressions.

Run:  python -m pytest bubble_detection/tests/ -q   (or just python this file)
"""
import os
import sys

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bubbles.prefix_ols import make_adf_engine, make_ar1_engine
from bubbles import psy
from bubbles.stability import rolling_cusum, rolling_chow


def _adf_ref(y, s, e, lags=1):
    yy = y[s:e + 1]
    dy = np.diff(yy)
    dep = dy[lags:]
    cols = [np.ones(len(yy) - 1 - lags), yy[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dy[lags - i:-i])
    return sm.OLS(dep, np.column_stack(cols)).fit().tvalues[1]


def test_adf_matches_statsmodels():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(600) * 0.01) + 5.0
    eng = make_adf_engine(y, lags=1)
    for s, e in [(0, 599), (100, 400), (250, 330), (500, 599)]:
        mine = eng.stats(np.array([s + 2]), np.array([e]))["tstat0"][0]
        assert np.isclose(mine, _adf_ref(y, s, e), atol=1e-8)


def test_ar1_ssr_matches_statsmodels():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(600) * 0.01) + 5.0
    eng = make_ar1_engine(y)
    yc = y - y.mean()
    for a, b in [(1, 599), (100, 400), (250, 330)]:
        dep = yc[a:b + 1]
        X = np.column_stack([np.ones(b - a + 1), yc[a - 1:b]])
        ref = sm.OLS(dep, X).fit().ssr
        assert np.isclose(eng.ssr(np.array([a]), np.array([b]))[0], ref, rtol=1e-8)


def test_bsadf_is_sup_of_window_adfs():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(600) * 0.01) + 5.0
    bs = psy.bsadf(y, min_window=40, max_window=200, lags=1, start_step=1)
    t = 300
    ref = max(_adf_ref(y, s, t) for s in range(t - 200 + 1, t - 40 + 2))
    assert np.isclose(bs[t], ref, atol=1e-8)


def test_bsadf_detects_explosive_root():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(600) * 0.01) + 5.0
    for i in range(400, 600):
        y[i] = 1.008 * y[i - 1] + rng.standard_normal() * 0.01
    bs = psy.bsadf(y, 40, 200, 1, 1)
    assert np.nanmax(bs[450:]) > 5.0


def test_cusum_and_chow_match_reference():
    rng = np.random.default_rng(1)
    n, W = 500, 336
    y = np.cumsum(rng.standard_normal(n) * 0.01) + 3.0
    yc = y - y.mean()
    s = n - W

    rls = sm.RecursiveLS(yc[s + 1:n], sm.add_constant(yc[s:n - 1])).fit()
    w_r = rls.resid_recursive[2 + 6:]                    # same warm-up as ours
    m = len(w_r)
    Wc = np.cumsum(w_r) / w_r.std(ddof=1)
    r = np.arange(1, m + 1)
    bound = 0.948 * (np.sqrt(m) + 2 * r / np.sqrt(m))
    ref = np.max(np.abs(Wc) / bound)
    c1, _ = rolling_cusum(y, window=W, warm=6)
    assert np.isclose(c1[n - 1], ref, rtol=1e-6)

    def ssr(d, X):
        return sm.OLS(d, X).fit().ssr
    obs_s, obs_e = s + 1, n - 1
    tau = (obs_s + obs_e) // 2
    full = ssr(yc[obs_s:obs_e + 1], sm.add_constant(yc[obs_s - 1:obs_e]))
    s1 = ssr(yc[obs_s:tau + 1], sm.add_constant(yc[obs_s - 1:tau]))
    s2 = ssr(yc[tau + 1:obs_e + 1], sm.add_constant(yc[tau:obs_e]))
    n0 = obs_e - obs_s + 1
    F = ((full - s1 - s2) / 2) / ((s1 + s2) / (n0 - 4))
    assert np.isclose(rolling_chow(y, window=W)[n - 1], F, rtol=1e-6)


if __name__ == "__main__":
    for fn in [test_adf_matches_statsmodels, test_ar1_ssr_matches_statsmodels,
               test_bsadf_is_sup_of_window_adfs, test_bsadf_detects_explosive_root,
               test_cusum_and_chow_match_reference]:
        fn()
        print(f"{fn.__name__}: OK")
    print("ALL TESTS PASS")
