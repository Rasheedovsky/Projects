"""Evaluation statistics on synthetic ground truth."""
import numpy as np

from spydt.evalx import (
    cscv_pbo,
    dm_test,
    dsr,
    effective_trials,
    hit_rate,
    mcnemar,
    psr,
    sharpe,
    strategy_net_returns,
)

RNG = np.random.default_rng(0)


def test_strategy_returns_costs_and_sign():
    prob = np.array([0.9, 0.1, 0.6])
    r = np.array([0.01, 0.01, -0.02])
    net = strategy_net_returns(prob, r, cost_per_side_bp=1.0)
    np.testing.assert_allclose(net, [0.01 - 2e-4, -0.01 - 2e-4, -0.02 - 2e-4])


def test_hit_rate():
    assert hit_rate(np.array([0.7, 0.3]), np.array([1.0, -1.0])) == 1.0


def test_psr_positive_strategy_high():
    r = RNG.normal(5e-4, 1e-3, 1000)  # strongly positive
    assert psr(r) > 0.99
    r0 = RNG.normal(0, 1e-3, 1000)
    assert 0.05 < psr(r0) < 0.95


def test_effective_trials_bounds():
    t = RNG.normal(0, 1, (500, 10))
    ind = effective_trials(t)
    assert 5 < ind <= 10  # independent columns -> near 10
    dup = np.repeat(t[:, :1], 10, axis=1) + RNG.normal(0, 1e-6, (500, 10))
    assert effective_trials(dup) < 1.5  # duplicated columns -> near 1


def test_dsr_deflates_vs_psr():
    r = RNG.normal(2e-4, 1e-3, 800)
    trials = RNG.normal(0, 1e-3, (800, 20))
    trials[:, 0] = r
    assert dsr(r, trials) <= psr(r) + 1e-9


def test_dm_detects_dominance():
    a = RNG.normal(8e-4, 1e-3, 800)
    b = RNG.normal(0, 1e-3, 800)
    t, p = dm_test(a, b)
    assert t > 2 and p < 0.05
    t2, p2 = dm_test(b, b + RNG.normal(0, 1e-6, 800))
    assert p2 > 0.05


def test_mcnemar_symmetric_null():
    h1 = RNG.random(500) > 0.5
    h2 = RNG.random(500) > 0.5
    _, p = mcnemar(h1, h2)
    assert p > 0.01


def test_cscv_pbo_overfit_vs_real():
    # one genuinely good trial among noise -> low PBO
    t = RNG.normal(0, 1e-3, (400, 12))
    t[:, 3] += 8e-4
    assert cscv_pbo(t, 8) < 0.4
    # pure noise -> PBO near 0.5
    noise = RNG.normal(0, 1e-3, (400, 12))
    assert 0.2 < cscv_pbo(noise, 8) < 0.8


def test_sharpe_scale():
    r = np.full(252, 4e-4) + RNG.normal(0, 1e-3, 252)
    assert 0 < sharpe(r) < 20
