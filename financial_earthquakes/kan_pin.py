"""
KAN-PIN: a physics-informed Kolmogorov-Arnold network that estimates the
parameters of the Hawkes/ETAS "financial earthquake" model.

This module combines the two ingredients the user supplied:

1. PINNverse (Almanstoetter, Vetter & Iber 2025, arXiv:2504.05248):
   parameter estimation in differential equations reformulated as
   *constrained* optimisation.  The data-fit term is the primary objective;
   the model equations (and parameter bounds) are explicit constraints,
   enforced with the Modified Differential Method of Multipliers (MDMM):
   gradient *descent* on network weights theta and model parameters eta,
   simultaneous gradient *ascent* on Lagrange multipliers, plus a quadratic
   penalty that makes constrained minima attractors.

2. PI-KAN (Kashefi & Mukerji 2025, "Physics-informed KAN PointNet"):
   replace the MLP inside a physics-informed network with a
   Kolmogorov-Arnold Network whose learnable activations are Jacobi
   polynomials: every edge (input k -> output j) carries its own function
   psi(z) = sum_i Lambda_i P_i^(a,b)(z), evaluated on tanh-squashed inputs
   so z lies in the Jacobi domain [-1, 1].

How a point process becomes "physics"
-------------------------------------
The ETAS model (see etas.py) says that, *given the observed event history*
{(t_i, m_i)}, the integrated intensity ("compensator")

    Lambda(t) = integral_0^t lambda_eta(s | H_s) ds

obeys
    dLambda/dt = lambda_eta(t),      Lambda(0) = 0,                  (DE)

where lambda_eta(t) = mu + K0 sum_{t_i<t} e^{alpha(m_i-M0)} phi(t-t_i) is a
known function of time once eta = (mu, K0, alpha, c, omega) is fixed.  The
*data* enter through the random time change theorem: if the model is right,
Lambda turns the events into a unit-rate Poisson process, so the counting
staircase N(t) is a noisy observation of Lambda(t) (at an event, the
unbiased target is i - 1/2).  Matching the network to the staircase is the
analogue of fitting noisy observations in PINNverse; the physics constraint
ties that fit back to the parameters eta we care about.

Design choices (and the failure modes that forced them)
-------------------------------------------------------
These were found the hard way; each guards against a concrete collapse mode
observed during development, all of which end in the degenerate homogeneous
Poisson solution (K0 = 0, mu = event rate) - the flat compensator satisfies
the physics *exactly* while losing only a little data fit, because the data
loss is dominated by intrinsic counting noise.  This is the concave-Pareto
pathology that motivates PINNverse, in point-process form:

* Integral (weak) form of the physics.  A Hawkes intensity spikes at every
  event and decays on the kernel scale c (< 1% of the window here); no
  smooth network satisfies the differential residual at that frequency
  (spectral bias).  We therefore constrain interval increments
      Lambda(s_{k+1}) - Lambda(s_k) = integral of lambda_eta over [s_k, s_{k+1}]
  whose right side is analytic for ETAS kernels.  Statistically this makes
  the estimator a 'compensator least squares' - a known consistent
  alternative to MLE for Hawkes processes.

* Fourier features + linear skip.  The KAN sees
  [x, sin(2 pi k x), cos(2 pi k x)], k = 1..F (PINNverse's own remedy for
  sharp solutions, and F must be large enough that the shortest wavelength
  T/F reaches the cluster scale), and the network is G(x) = x + KAN(x): the
  skip term is the exact Poisson compensator, so the KAN only learns the
  *deviation from Poisson* - the clustering.

* Depth-1 KAN => two exact linear solves.  A single KAN layer is linear in
  its coefficients, so fitting the network to given targets is an ordinary
  ridge least squares.  This kills the optimisation pathologies of the
  curriculum's first stages (Adam stalls on the weak high-frequency
  gradients of the staircase residual) while remaining a genuine KAN:
  edge-wise learnable Jacobi activations, exactly the Kolmogorov-Arnold
  inner sum of the PI-KAN paper.

The four-phase curriculum
-------------------------
  A. Staircase solve: ridge least squares of the KAN on the counting
     staircase (the data), giving the empirical compensator.
  B. Parameter calibration: freeze the network; fit eta by nonlinear least
     squares of the analytic integrated intensity against the network's
     increments (coarse grid over (K0, c) + Adam polish - 5 parameters,
     well-conditioned).
  B2. Physics projection: freeze eta; re-solve the KAN on the *analytic*
     compensator Lambda_eta (again exact ridge LS), projecting the
     step-overfit interpolant onto the manifold of exact ETAS compensators.
     After this step the constraints are (nearly) satisfied AND eta is
     informative.
  C. MDMM negotiation (the PINNverse step): joint Adam descent on
     (theta, eta) with gradient ascent on the multipliers of the DE/IC
     constraints and of the parameter bounds, letting data and physics
     settle on the balanced Pareto point.

Why bother, when MLE exists?  On a clean univariate Hawkes, MLE is the
efficient estimator and should win - we use it as the benchmark.  The
KAN-PIN route matters where the thesis's later chapters go: models whose
likelihood is intractable or expensive (non-affine Hawkes, option-implied
estimation), where a physics-informed surrogate with constraints can still
be trained.  Here it doubles as an independent cross-check of the MLE.

Known limitation (quantified in the synthetic validation): the kernel time
scale c is biased SHORT.  The smooth interpolant spreads each event's own
+1 staircase step into the immediately following intervals, which the
physics fit can only explain as near-instant aftershocks ("interpolation
leakage").  MLE is immune because an event's kernel contribution starts
strictly after its own arrival time.  The background rate mu and the
branching ratio K0 - the quantities that carry the economic content
(criticality, cluster sizes) - are recovered well.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.flatten import flatten

# --------------------------------------------------------------------------
# Jacobi-polynomial KAN layer (PI-KAN building block)
# --------------------------------------------------------------------------

def jacobi_basis(z, degree: int, a: float = 0.0, b: float = 0.0):
    """Stack of Jacobi polynomials P_0..P_degree evaluated at z (any shape).

    Standard three-term recurrence; a = b = 0 gives Legendre polynomials.
    Returns shape z.shape + (degree+1,).
    """
    P = [anp.ones_like(z)]
    if degree >= 1:
        P.append(0.5 * (a - b + (a + b + 2.0) * z))
    for n in range(2, degree + 1):
        c1 = 2.0 * n * (n + a + b) * (2.0 * n + a + b - 2.0)
        c2 = (2.0 * n + a + b - 1.0) * (a * a - b * b)
        c3 = (2.0 * n + a + b - 1.0) * (2.0 * n + a + b) * (2.0 * n + a + b - 2.0)
        c4 = 2.0 * (n + a - 1.0) * (n + b - 1.0) * (2.0 * n + a + b)
        P.append(((c2 + c3 * z) * P[n - 1] - c4 * P[n - 2]) / c1)
    return anp.stack(P, axis=-1)


# --------------------------------------------------------------------------
# The KAN-PIN estimator for ETAS parameters
# --------------------------------------------------------------------------

class KANPINHawkes:
    """Estimate eta = (mu, K0, alpha, c, omega) of the ETAS model with a
    physics-informed KAN trained by the four-phase curriculum described in
    the module docstring (ending in PINNverse's MDMM).

    Parameters
    ----------
    events    : EventData from etas.py (times, mags, M0, T)
    kernel    : 'power' or 'exp' triggering kernel (matches etas.py)
    use_alpha : include the magnitude-leverage parameter alpha
    degree    : Jacobi polynomial degree of the KAN edge activations
    n_fourier : number of Fourier feature pairs; the shortest representable
                wavelength is T/n_fourier, which must reach the cluster
                time scale for the clustering signal to survive
    """

    PARAM_NAMES = ("mu", "K0", "alpha", "c", "omega")

    def __init__(self, events, kernel: str = "exp", use_alpha: bool = False,
                 degree: int = 5, jacobi_ab=(0.0, 0.0),
                 n_colloc: int = 400, n_fourier: int = 32, seed: int = 0):
        self.ev = events
        self.kernel = kernel
        self.use_alpha = use_alpha
        self.degree = degree
        self.a, self.b = jacobi_ab
        self.n_fourier = n_fourier
        self.N, self.T = events.n, events.T
        self.rho = self.N / self.T                  # mean event rate

        # ---- parameter bounds (constraints, PINNverse-style) ----
        self.lower = np.array([1e-6, 1e-6, 0.0, 1e-2, 0.10])
        self.upper = np.array([5 * self.rho, 0.995,
                               2.0 / max(events.mags.std(), 1e-4),
                               self.T / 10, 10.0])

        # ---- collocation NODES for the integral-form physics: a uniform
        # grid refined with points just after each event (the post-event
        # decay is what identifies K0 and c) ----
        base = np.linspace(0.0, self.T, n_colloc)
        near = (events.times[:, None] + np.array([0.0, 1.0, 4.0, 12.0])).ravel()
        self.s_nodes = np.unique(np.clip(np.concatenate([base, near]), 0.0, self.T))

        # ---- data observations: the scaled counting staircase N(t)/N at
        # the nodes; AT an event the unbiased target is i - 1/2 (time
        # rescaling), hence the left/right average ----
        self.t_data = self.s_nodes
        left = np.searchsorted(events.times, self.t_data, side="left")
        right = np.searchsorted(events.times, self.t_data, side="right")
        self.y_data = 0.5 * (left + right) / self.N

        # ---- network: depth-1 KAN over Fourier features, linear skip ----
        d_feat = 2 * n_fourier + 1
        rng = np.random.default_rng(seed)
        Lam0 = rng.normal(0.0, 1e-3, size=(d_feat, 1, degree + 1))
        eta0 = np.array([self.rho, 0.3,
                         (10.0 if use_alpha else 0.0), self.T / 50, 1.0])
        self.params = {"layers": [Lam0], "eta": eta0}
        # one multiplier per equality constraint (DE, IC) + one per bound
        self.mult = {"lam": np.zeros(2), "chi": np.zeros(5)}
        self.history: dict[str, list] = {k: [] for k in
                                         ("L_data", "L_de", "L_ic", *self.PARAM_NAMES)}

    # ------------- network pieces (autograd-differentiable) -------------

    def _features(self, t):
        x = anp.reshape(t, (-1, 1)) / self.T
        k = anp.arange(1.0, self.n_fourier + 1.0)[None, :]
        return x, anp.concatenate(
            [x, anp.sin(2.0 * anp.pi * x * k), anp.cos(2.0 * anp.pi * x * k)],
            axis=1)

    def _G(self, layers, t):
        """Scaled compensator: Lambda_theta(t) = N * G(t/T), with
        G(x) = x + KAN(features(x)) - KAN(features(0)).

        G(x) = x is the exact compensator of a homogeneous Poisson process
        with the empirical rate, so the KAN only has to learn the deviation
        from Poisson (the clustering).  Subtracting the t = 0 output
        hard-encodes the initial condition Lambda(0) = 0 into the
        architecture (as PINNverse does for exactly-fulfilled IC/BCs),
        which is far more robust than enforcing it through a multiplier."""
        tt = anp.concatenate([anp.array([0.0]), anp.reshape(t, (-1,))])
        x, feats = self._features(tt)
        z = anp.tanh(feats)                              # (B+1, d_feat)
        basis = jacobi_basis(z, self.degree, self.a, self.b)
        out = anp.einsum("bik,iok->bo", basis, layers[0])[:, 0]
        return x[1:, 0] + out[1:] - out[0]

    def _design_matrix(self, t):
        """The depth-1 KAN is linear in its coefficients:
        G(x) = x + Phi(t) @ vec(Lambda).  Returns Phi, shape (B, d_feat*(deg+1))."""
        _, feats = self._features(t)
        z = np.tanh(np.asarray(feats))
        basis = np.asarray(jacobi_basis(z, self.degree, self.a, self.b))
        B = basis.shape[0]
        return basis.reshape(B, -1)

    def _solve_layers(self, t, y, ridge: float = 1e-6):
        """Exact ridge least-squares fit of the KAN coefficients to targets
        y at times t (used by phases A and B2).  Because the architecture
        subtracts the t = 0 output (hard-encoded IC), the effective design
        matrix is Phi(t) - Phi(0)."""
        t = np.asarray(t, dtype=float)
        Phi = self._design_matrix(t) - self._design_matrix(np.array([0.0]))
        resid = np.asarray(y) - t / self.T
        A = Phi.T @ Phi + ridge * len(t) * np.eye(Phi.shape[1])
        coef = np.linalg.solve(A, Phi.T @ resid)
        d_feat = 2 * self.n_fourier + 1
        return [coef.reshape(d_feat, 1, self.degree + 1)]

    # ------------- physics pieces -------------

    def _integrated_lambda(self, eta, a, b):
        """integral_a^b lambda_eta(s) ds, analytic via the kernel CDF:
        mu (b-a) + K0 sum_i boost_i [PHI(b - t_i) - PHI(a - t_i)]."""
        mu, K0, alpha, c, omega = eta[0], eta[1], eta[2], eta[3], eta[4]
        if not self.use_alpha:
            alpha = 0.0
        boost = anp.exp(alpha * (self.ev.mags - self.ev.M0))[None, :]
        def PHI(u):
            up = anp.where(u > 0, u, 0.0)
            if self.kernel == "power":
                return 1.0 - (1.0 + up / c) ** (-omega)
            return 1.0 - anp.exp(-up / c)
        da = a[:, None] - self.ev.times[None, :]
        db = b[:, None] - self.ev.times[None, :]
        return mu * (b - a) + K0 * anp.sum(boost * (PHI(db) - PHI(da)), axis=1)

    def _losses(self, params):
        """The three loss components of the PINNverse formulation."""
        layers, eta = params["layers"], params["eta"]
        # data: staircase observations over the whole domain
        Gd = self._G(layers, self.t_data)
        L_data = anp.mean((Gd - self.y_data) ** 2)
        # physics in integral (weak) form on collocation intervals,
        # each residual normalised by the expected count in the interval
        s = self.s_nodes
        Gs = self._G(layers, s)
        ds = s[1:] - s[:-1]
        incr_net = (Gs[1:] - Gs[:-1]) * self.N
        incr_phys = self._integrated_lambda(eta, s[:-1], s[1:])
        scale = self.rho * np.maximum(ds, self.T / (4.0 * len(ds)))
        L_de = anp.mean(((incr_net - incr_phys) / scale) ** 2)
        # initial condition Lambda(0) = 0
        L_ic = self._G(layers, anp.array([0.0]))[0] ** 2
        return L_data, L_de, L_ic

    def _infeasibility(self, eta):
        """V_j = clip(eta_j) - eta_j  (PINNverse bound constraints)."""
        return anp.clip(eta, self.lower, self.upper) - eta

    def _augmented_lagrangian(self, params, lam, chi, penalty=1.0):
        L_data, L_de, L_ic = self._losses(params)
        V = self._infeasibility(params["eta"])
        return (L_data
                + lam[0] * L_de + 0.5 * penalty * L_de ** 2
                + lam[1] * L_ic + 0.5 * penalty * L_ic ** 2
                + anp.sum(chi * V) + 0.5 * penalty * anp.sum(V ** 2))

    # ------------- phase B: calibrate eta against the frozen network ----

    def _calibrate_eta(self, params, adam_steps: int = 400) -> np.ndarray:
        """With the network frozen, choose eta to satisfy the physics: an
        ordinary nonlinear least squares in 5 parameters.  A coarse grid
        over (K0, c) - with mu tied through the stationarity relation
        rate = mu / (1 - K0) - finds the right valley; Adam polishes."""
        layers = params["layers"]
        s = self.s_nodes
        Gs = np.asarray(self._G(layers, s))
        incr_net = (Gs[1:] - Gs[:-1]) * self.N
        ds = s[1:] - s[:-1]
        scale = self.rho * np.maximum(ds, self.T / (4.0 * len(ds)))

        def L_de_of(eta):
            incr_phys = self._integrated_lambda(eta, s[:-1], s[1:])
            return anp.mean(((incr_net - incr_phys) / scale) ** 2)

        best, best_val = None, np.inf
        for K0 in (0.05, 0.2, 0.4, 0.6, 0.8):
            for c in np.geomspace(1.0, self.T / 12, 8):
                eta = np.array([max(self.rho * (1 - K0), self.lower[0]), K0,
                                params["eta"][2], c, params["eta"][4]])
                val = float(L_de_of(eta))
                if val < best_val:
                    best, best_val = eta, val
        g_eta = grad(L_de_of)
        eta = best.copy()
        m = np.zeros_like(eta); v = np.zeros_like(eta)
        for k in range(1, adam_steps + 1):
            g = np.asarray(g_eta(eta))
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            eta = eta - 2e-2 * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
            eta = np.clip(eta, self.lower, self.upper)
        return eta

    # ------------- the full curriculum -------------

    def fit(self, epochs: int = 800, lr: float = 2e-3, lr_end: float = 2e-4,
            penalty: float = 1.0, ridge: float = 1e-4, verbose: bool = True):
        """Run phases A (staircase LS solve), B (eta calibration),
        B2 (physics-projection LS solve), then `epochs` of MDMM (phase C)."""
        # ---- phase A: exact LS fit of the KAN to the staircase ----
        self.params["layers"] = self._solve_layers(self.t_data, self.y_data, ridge)
        L = self._losses(self.params)
        if verbose:
            print(f"  phase A  (staircase solve) L_data={L[0]:.3e}")
        # ---- phase B: calibrate eta on the frozen network ----
        self.params["eta"] = self._calibrate_eta(self.params)
        if verbose:
            print(f"  phase B  (eta calibration) eta={np.round(self.params['eta'], 4)}")
        # ---- phase B2: project the network onto the physics manifold ----
        s = self.s_nodes
        Lam_eta = np.concatenate([[0.0], np.cumsum(np.asarray(
            self._integrated_lambda(self.params["eta"], s[:-1], s[1:])))])
        self.params["layers"] = self._solve_layers(s, Lam_eta / self.N, ridge)
        L = self._losses(self.params)
        if verbose:
            print(f"  phase B2 (physics projection) L_data={L[0]:.3e} L_de={L[1]:.3e}")

        # ---- phase C: MDMM negotiation (PINNverse update rule) ----
        flat, unflatten = flatten(self.params)
        lam, chi = self.mult["lam"].copy(), self.mult["chi"].copy()

        def objective(flat_params, lam_, chi_):
            return self._augmented_lagrangian(unflatten(flat_params), lam_, chi_, penalty)

        gfun = grad(objective, 0)
        m = np.zeros_like(flat); v = np.zeros_like(flat)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in range(1, epochs + 1):
            alpha_k = lr * (lr_end / lr) ** (k / epochs)
            g = gfun(flat, lam, chi)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            flat = flat - alpha_k * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps)
            p = unflatten(flat)
            L_data, L_de, L_ic = self._losses(p)
            V = self._infeasibility(p["eta"])
            lam = lam + alpha_k * np.array([L_de, L_ic])      # multiplier ascent
            chi = chi + alpha_k * np.asarray(V)
            if k % 10 == 0 or k == 1:
                self.history["L_data"].append(float(L_data))
                self.history["L_de"].append(float(L_de))
                self.history["L_ic"].append(float(L_ic))
                eta_c = np.clip(np.asarray(p["eta"]), self.lower, self.upper)
                for nme, val in zip(self.PARAM_NAMES, eta_c):
                    self.history[nme].append(float(val))
            if verbose and (k % max(epochs // 4, 1) == 0 or k == 1):
                print(f"  phase C  epoch {k:5d}  L_data={L_data:.3e}"
                      f"  L_de={L_de:.3e}  L_ic={L_ic:.3e}"
                      f"  eta={np.round(np.asarray(p['eta']), 4)}")
        self.params = unflatten(flat)
        self.mult = {"lam": lam, "chi": chi}
        return self

    # ------------- results -------------

    @property
    def eta_hat(self) -> dict:
        eta = np.clip(np.asarray(self.params["eta"]), self.lower, self.upper)
        out = dict(zip(self.PARAM_NAMES, (float(x) for x in eta)))
        if not self.use_alpha:
            out["alpha"] = 0.0
        if self.kernel == "exp":
            out["omega"] = float("nan")
        return out

    def compensator_curve(self, n: int = 400):
        """(t, Lambda_theta(t)) for plotting against the event staircase."""
        t = np.linspace(0.0, self.T, n)
        return t, self.N * np.asarray(self._G(self.params["layers"], t))
