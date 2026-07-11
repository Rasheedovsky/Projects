"""Mono-LPPLS-NN (M-LNN), paper Sec. 2.1.

A fresh 2-hidden-layer ReLU network is trained per time series (Eq. 2 /
Fig. 1). The network maps the min-max-scaled series X (length n) to the three
nonlinear parameters (tc, m, w); the four linear parameters are recovered
analytically via Eq. 8 inside the loss, and the training loss is

    L = MSE(X, LPPLS(Y_hat)) + L_penalty          (Sec. 2.1.1)
    L_penalty = alpha * sum_k [max(0, th_min - th) + max(0, th - th_max)]

with bounds tc in [t2 - 0.2*t2, t2 + 0.2*t2] (= [0.8, 1.2] since t2 = 1),
m in [0.1, 1], w in [6, 13]. Adam, lr = 1e-2, best state kept (Sec. 2.1.2).

Choices the paper leaves unspecified (documented in the report):
  hidden widths 128/128, alpha = 10, 1500 epochs, output-layer bias
  initialised at the centre of the admissible box so training starts from an
  in-bounds parameter guess.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

EPS = 1e-8
BOUNDS = jnp.array([[0.8, 1.2], [0.1, 1.0], [6.0, 13.0]])  # tc, m, w
INIT_OUT = jnp.array([1.05, 0.5, 9.5])
HIDDEN = 128
ALPHA = 10.0
LR = 1e-2
EPOCHS = 1500


def _init_params(key, n_in, hidden=HIDDEN):
    k1, k2, k3 = jax.random.split(key, 3)
    def dense(k, n, m):
        return jax.random.normal(k, (n, m)) * jnp.sqrt(2.0 / n)
    return dict(
        W1=dense(k1, n_in, hidden), b1=jnp.zeros(hidden),
        W2=dense(k2, hidden, hidden), b2=jnp.zeros(hidden),
        Wo=dense(k3, hidden, 3) * 0.01, bo=INIT_OUT,
    )


def _forward(params, x):
    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])
    return h2 @ params["Wo"] + params["bo"]  # (tc, m, w)


def _lppls_mse(theta, t, x):
    tc, m, w = theta[0], theta[1], theta[2]
    dt = jnp.abs(tc - t) + EPS
    logdt = jnp.log(dt)
    f = dt**m
    g = f * jnp.cos(w * logdt)
    h = f * jnp.sin(w * logdt)
    M = jnp.stack([jnp.ones_like(t), f, g, h], axis=1)
    beta = jnp.linalg.solve(M.T @ M + EPS * jnp.eye(4), M.T @ x)
    resid = x - M @ beta
    return jnp.mean(resid**2)


def _penalty(theta):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return jnp.sum(jnp.maximum(0.0, lo - theta) + jnp.maximum(0.0, theta - hi))


def _loss(params, t, x):
    theta = _forward(params, x)
    return _lppls_mse(theta, t, x) + ALPHA * _penalty(theta), theta


@partial(jax.jit, static_argnames=())
def _step(params, opt_state, t, x):
    (loss, theta), grads = jax.value_and_grad(_loss, has_aux=True)(params, t, x)
    updates, opt_state = _OPT.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, theta


_OPT = optax.adam(LR)


def fit_mlnn(x, epochs=EPOCHS, seed=0):
    """Train a fresh M-LNN on one min-max-scaled series x (any length n).
    Returns dict with tc, m, w (normalised time units) and best total loss."""
    n = len(x)
    t = jnp.linspace(0.0, 1.0, n)
    x = jnp.asarray(x, dtype=jnp.float32)
    params = _init_params(jax.random.PRNGKey(seed), n)
    opt_state = _OPT.init(params)
    best_loss, best_theta = np.inf, None
    for _ in range(epochs):
        params, opt_state, loss, theta = _step(params, opt_state, t, x)
        loss = float(loss)
        if loss < best_loss:
            best_loss, best_theta = loss, np.asarray(theta)
    tc, m, w = (float(v) for v in best_theta)
    return dict(tc=tc, m=m, w=w, loss=best_loss)
