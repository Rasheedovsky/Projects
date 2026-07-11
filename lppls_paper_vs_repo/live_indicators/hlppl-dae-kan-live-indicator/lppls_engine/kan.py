"""M-LNN-KAN: the Mono-LPPLS-NN with Kolmogorov-Arnold Network layers
(Liu et al. 2024, arXiv:2404.19756) replacing the ReLU MLP of paper Sec. 2.1.

Everything except the network parameterisation is IDENTICAL to mlnn.py —
same PINN-style loss (MSE of the reconstructed LPPLS series via the analytic
linear solve, Eq. 8), same bound penalty, same Adam(1e-2), same best-state
selection — so any accuracy/time difference is attributable to the
KAN-vs-MLP change alone.

KAN layer: y_j = sum_i [ w_base[i,j] * silu(x_i) + sum_k c[i,j,k] * B_k(x~_i) ]
with cubic B-spline basis B_k on a fixed uniform grid over [-1, 1]
(GRID_SIZE intervals -> GRID_SIZE + 3 basis functions) and x~ = tanh(x) to
keep activations inside the grid (fixed-grid variant, cf. efficient-kan).
Widths [32, 32] follow KAN convention of narrower layers than MLPs; the
output layer stays linear with the same bias init as the MLP version.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .mlnn import ALPHA, BOUNDS, INIT_OUT, LR, EPOCHS, _lppls_mse, _penalty

GRID_SIZE = 8
SPLINE_ORDER = 3
N_BASIS = GRID_SIZE + SPLINE_ORDER
HIDDEN = 32

# uniform extended knot vector over [-1, 1]
_H = 2.0 / GRID_SIZE
_KNOTS = jnp.arange(-SPLINE_ORDER, GRID_SIZE + SPLINE_ORDER + 1) * _H - 1.0  # len G+2k+1


def bspline_basis(u):
    """Cox-de Boor recursion, u shape (...,) -> basis shape (..., N_BASIS)."""
    u = u[..., None]  # (..., 1)
    t = _KNOTS
    # order 0: indicator on knot spans (len(t)-1 functions)
    B = jnp.where((u >= t[:-1]) & (u < t[1:]), 1.0, 0.0)
    for k in range(1, SPLINE_ORDER + 1):
        left = (u - t[: -(k + 1)]) / (t[k:-1] - t[: -(k + 1)]) * B[..., :-1]
        right = (t[k + 1 :] - u) / (t[k + 1 :] - t[1:-k]) * B[..., 1:]
        B = left + right
    return B  # (..., N_BASIS)


def _init_kan_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    return dict(
        base=jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
        coef=jax.random.normal(k2, (n_in, n_out, N_BASIS)) * (0.1 / jnp.sqrt(n_in)),
    )


def _kan_layer(layer, x):
    xb = jnp.tanh(x)
    B = bspline_basis(xb)                      # (n_in, N_BASIS)
    spline = jnp.einsum("ik,iok->o", B, layer["coef"])
    base = jax.nn.silu(x) @ layer["base"]
    return base + spline


def _init_params(key, n_in, hidden=HIDDEN):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        l1=_init_kan_layer(k1, n_in, hidden),
        l2=_init_kan_layer(k2, hidden, hidden),
        Wo=jax.random.normal(k3, (hidden, 3)) * 0.01,
        bo=INIT_OUT,
    )


def _forward(params, x):
    h1 = _kan_layer(params["l1"], x)
    h2 = _kan_layer(params["l2"], h1)
    return h2 @ params["Wo"] + params["bo"]


def _loss(params, t, x):
    theta = _forward(params, x)
    return _lppls_mse(theta, t, x) + ALPHA * _penalty(theta), theta


_OPT = optax.adam(LR)


@partial(jax.jit, static_argnames=())
def _step(params, opt_state, t, x):
    (loss, theta), grads = jax.value_and_grad(_loss, has_aux=True)(params, t, x)
    updates, opt_state = _OPT.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, theta


def fit_mlnn_kan(x, epochs=EPOCHS, seed=0):
    """Train a fresh M-LNN-KAN on one min-max-scaled series x.
    Same interface/protocol as mlnn.fit_mlnn."""
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
