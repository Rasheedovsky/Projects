"""Implementation of "Deep LPPLS: Forecasting of temporal critical points in
natural, engineering and financial systems" (Nielsen, Sornette, Raissi;
arXiv:2405.12803), built to compare against the reference implementation in
Boulder-Investment-Technologies/lppls.

Modules
-------
core        LPPLS function, linear-parameter solve (Eq. 4-8 of the paper)
synthetic   Synthetic LPPLS series generation with white/AR(1) noise (Table 1)
lm          Paper's benchmark calibration: Levenberg-Marquardt with multistart
mlnn        Mono-LPPLS-NN (M-LNN, Sec. 2.1) in JAX
plnn        Poly-LPPLS-NN (P-LNN, Sec. 2.2) in JAX
"""
