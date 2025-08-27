# src/kernels.py
# Acceleration kernels K(r;L) and their integrated potential kernels U(r;L).

import numpy as np

# ---- acceleration kernels (as in the paper) ----
def plummer_kernel(r, L):
    return 1.0 / (4.0 * np.pi * (r**2 + L**2))

def exp_core_kernel(r, L):
    return np.exp(-r / L) / (4.0 * np.pi * (r**2 + L**2))

# aliases (optional, handy)
K_plummer  = plummer_kernel
K_exp_core = exp_core_kernel

# ---- potential kernels (for convolution, then take ∇) ----
def U_plummer(r, L):
    """U(r;L) = (1/(4π L)) * arctan(r/L), with U(0)=0."""
    L = np.clip(L, 1e-9, None)
    return (1.0 / (4.0 * np.pi * L)) * np.arctan(r / L)

def U_exp_core(r, L, rmax=None, n=4096):
    """
    Build U by integrating K_exp_core from 0..r (fast, vectorized).
    Returns an array matching the shape of r.
    """
    r = np.asarray(r)
    if rmax is None:
        rmax = float(np.max(r))
    rt = np.linspace(0.0, rmax, n)
    K = exp_core_kernel(rt, L)
    Utab = np.zeros_like(rt)
    Utab[1:] = np.cumsum(0.5 * (K[1:] + K[:-1]) * (rt[1:] - rt[:-1]))
    return np.interp(r, rt, Utab)
