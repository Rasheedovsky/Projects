"""Denoising autoencoder (DAE) for price-window cleaning.

Unlike SSA (structure-agnostic), the DAE is trained on synthetic
(noisy -> clean) LPPLS pairs from the same generator used for the P-LNN
(white + AR(1) mixture, Table 1 amplitudes), so it learns the LPPLS signal
prior. Architecture: dense 252 -> 128 -> 64 -> 128 -> 252, ReLU hidden
activations, linear output.

Loss: MSE with an endpoint weight ramp (1 -> 3 over the last 30 points).
The SSA experiment showed that endpoint distortion is what corrupts t_c
estimates; the ramp makes the reconstruction prioritise fidelity where the
critical-time information lives. Inputs/outputs are min-max-scaled windows
(same convention as every other model here).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

N_IN = 252
SIZES = [N_IN, 128, 64, 128, N_IN]
LR = 1e-3
EPOCHS = 30
BATCH = 256

_W = np.ones(N_IN)
_W[-30:] = np.linspace(1.0, 3.0, 30)
ENDPOINT_WEIGHT = jnp.asarray(_W)


def init_params(key):
    keys = jax.random.split(key, len(SIZES) - 1)
    params = {}
    for i, (k, (a, b)) in enumerate(zip(keys, zip(SIZES[:-1], SIZES[1:])), start=1):
        params[f"W{i}"] = jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a)
        params[f"b{i}"] = jnp.zeros(b)
    return params


def forward(params, x):
    h = x
    n_layers = len(SIZES) - 1
    for i in range(1, n_layers):
        h = jax.nn.relu(h @ params[f"W{i}"] + params[f"b{i}"])
    return h @ params[f"W{n_layers}"] + params[f"b{n_layers}"]


def _loss(params, xb, yb):
    rec = forward(params, xb)
    return jnp.mean(ENDPOINT_WEIGHT * (rec - yb) ** 2)


def train(X_noisy, X_clean, Xv_noisy, Xv_clean, epochs=EPOCHS, seed=0, log=print):
    opt = optax.adam(LR)
    params = init_params(jax.random.PRNGKey(seed))
    opt_state = opt.init(params)
    n = (len(X_noisy) // BATCH) * BATCH
    Xv_noisy, Xv_clean = jnp.asarray(Xv_noisy), jnp.asarray(Xv_clean)

    @jax.jit
    def epoch_scan(params, opt_state, Xb, yb):
        def step(carry, xy):
            params, opt_state = carry
            xb, yb = xy
            loss, grads = jax.value_and_grad(_loss)(params, xb, yb)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), (Xb, yb))
        return params, opt_state, jnp.mean(losses)

    @jax.jit
    def val_loss(params):
        return _loss(params, Xv_noisy, Xv_clean)

    rng = np.random.default_rng(seed)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        idx = rng.permutation(len(X_noisy))[:n]
        Xb = jnp.asarray(X_noisy[idx].reshape(-1, BATCH, N_IN))
        yb = jnp.asarray(X_clean[idx].reshape(-1, BATCH, N_IN))
        params, opt_state, tr = epoch_scan(params, opt_state, Xb, yb)
        vl = val_loss(params)
        hist["train"].append(float(tr))
        hist["val"].append(float(vl))
        if (ep + 1) % 5 == 0:
            log(f"  epoch {ep + 1:2d}/{epochs}  train {float(tr):.6f}  val {float(vl):.6f}")
    return params, hist


def save_params(params, path):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path):
    with np.load(path) as f:
        return {k: jnp.asarray(f[k]) for k in f.files}


_fwd_jit = jax.jit(forward)


def clean(params, x_scaled):
    """Denoise one min-max-scaled 252-point window."""
    x = jnp.asarray(np.atleast_2d(x_scaled), dtype=jnp.float32)
    return np.asarray(_fwd_jit(params, x))[0]
