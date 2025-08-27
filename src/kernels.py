import numpy as np

# --- Primary names (simple) ---
def plummer_kernel(r, L):
    """
    Plummer acceleration kernel:
        K_Plummer(r; L) = 1 / [4π (r^2 + L^2)]
    r : float or np.ndarray (kpc)
    L : float (kpc), a positive global length scale
    """
    return 1.0 / (4.0 * np.pi * (r**2 + L**2))

def exp_core_kernel(r, L):
    """
    Exponentially softened core kernel:
        K_exp-core(r; L) = exp(-r/L) / [4π (r^2 + L^2)]
    r : float or np.ndarray (kpc)
    L : float (kpc)
    """
    return np.exp(-r / L) / (4.0 * np.pi * (r**2 + L**2))

# --- Alias names (match the paper/skeleton later) ---
K_plummer   = plummer_kernel
K_exp_core  = exp_core_kernel
