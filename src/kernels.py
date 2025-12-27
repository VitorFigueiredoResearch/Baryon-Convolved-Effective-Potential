import numpy as np

def U_plummer(r, L):
    """Legacy Plummer (1/r)"""
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    return 1.0 / np.sqrt(r_safe**2 + L**2)

def U_ananta_hybrid(r, L, **kwargs):
    """
    Shape-only Ananta hybrid kernel. No amplitude normalization here.
    """
    beta = kwargs.get("beta", 1.0)  # keep for compatibility but do NOT use for amplitude
    L = float(L)
    eps = 0.01 * L
    r_safe = np.sqrt(r**2 + eps**2)
    U = 0.5 * np.log(1.0 + (r_safe / L)**2)
    # return shape-only (beta parameter ignored for amplitude; allow callers to multiply)
    return U.astype(np.float64)


def U_exp_core(r, L):
    """Legacy Exponential"""
    L = float(L)
    return np.exp(-r/L) / (r + 1e-6)
