"""Poly-LPPLS-NN (P-LNN), paper Sec. 2.2.

Architecture (Eq. 3 / Fig. 2): input 252 -> 4 hidden ReLU layers -> 3 outputs
(tc, m, w). Trained supervised on synthetic noisy LPPLS series with loss =
MSE between predicted and true (tc, m, w). Paper training spec (Sec. 2.2.4):
20 epochs, batch size 8, Adam, lr = 1e-5, min-max-scaled inputs.

Hidden widths are not given in the paper; we use 256 per layer (documented).

Three variants (Sec. 2.2.1 / Table 1):
  P-LNN-100K       white noise, amplitude U(0.01, 0.15)
  P-LNN-100K-AR1   AR(1) noise (phi = 0.9), amplitude U(0.01, 0.05)
  P-LNN-100K-BOTH  50/50 mix of the two
Each trains on 100,000 series with 33,333 validation series.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

N_IN = 252
HIDDEN = 256
LR = 1e-5
EPOCHS = 20
BATCH = 8


def init_params(key, n_in=N_IN, hidden=HIDDEN):
    keys = jax.random.split(key, 5)
    sizes = [(n_in, hidden), (hidden, hidden), (hidden, hidden), (hidden, hidden), (hidden, 3)]
    params = {}
    for i, (k, (a, b)) in enumerate(zip(keys, sizes), start=1):
        params[f"W{i}"] = jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a)
        params[f"b{i}"] = jnp.zeros(b)
    # start predictions at the centre of the label box (tc~1.1, m~0.5, w~9.5)
    params["b5"] = jnp.array([1.1, 0.5, 9.5])
    return params


def forward(params, x):
    h = x
    for i in range(1, 5):
        h = jax.nn.relu(h @ params[f"W{i}"] + params[f"b{i}"])
    return h @ params["W5"] + params["b5"]


def _loss(params, xb, yb):
    pred = forward(params, xb)
    return jnp.mean((pred - yb) ** 2)


def train(X, y, X_val, y_val, epochs=EPOCHS, batch=BATCH, lr=LR, seed=0, log=print):
    """Train a P-LNN variant. Returns (params, history dict)."""
    opt = optax.adam(lr)
    params = init_params(jax.random.PRNGKey(seed))
    opt_state = opt.init(params)

    n = (len(X) // batch) * batch
    X_val = jnp.asarray(X_val)
    y_val = jnp.asarray(y_val)

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
        return _loss(params, X_val, y_val)

    rng = np.random.default_rng(seed)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        idx = rng.permutation(len(X))[:n]
        Xb = jnp.asarray(X[idx].reshape(-1, batch, X.shape[1]))
        yb = jnp.asarray(y[idx].reshape(-1, batch, 3))
        params, opt_state, tr = epoch_scan(params, opt_state, Xb, yb)
        vl = val_loss(params)
        hist["train"].append(float(tr))
        hist["val"].append(float(vl))
        log(f"  epoch {ep + 1:2d}/{epochs}  train {float(tr):.5f}  val {float(vl):.5f}")
    return params, hist


def save_params(params, path):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path):
    with np.load(path) as f:
        return {k: jnp.asarray(f[k]) for k in f.files}


_predict_jit = jax.jit(forward)


def predict(params, X):
    """Predict (tc, m, w) for min-max-scaled series X of shape (k, 252)."""
    X = jnp.asarray(np.atleast_2d(X), dtype=jnp.float32)
    return np.asarray(_predict_jit(params, X))
