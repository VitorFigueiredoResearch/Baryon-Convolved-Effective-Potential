import numpy as np

def U_plummer(r, L):
    """Legacy Plummer (1/r)"""
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    return 1.0 / np.sqrt(r_safe**2 + L**2)

def U_ananta_hybrid(r, L, **kwargs):
    """
    R3-6: The Corrected Ananta Potential.
    
    Physics:
    - Form: Logarithmic Potential ~ ln(r).
    - Force: Decays as 1/r (giving Flat Rotation Curves).
    - Units: Multiplied by (1/L) to give correct [1/kpc] dimensions.
    """

    # --- NEW: global amplitude correction (β) ---
    beta = kwargs.get("beta", 1.0)

    L = float(L)
    # Softening length (core size)
    eps = 0.01 * L 
    r_safe = np.sqrt(r**2 + eps**2)
    
    # Dimensional Fix: Divide by L to get units of 1/kpc
    # Shape Fix: Logarithmic growth for long-range memory
    U = (1.0 / L) * 0.5 * np.log(1.0 + (r_safe / L)**2)

    # --- NEW: apply β here ---
    return beta * U

def U_exp_core(r, L):
    """Legacy Exponential"""
    L = float(L)
    return np.exp(-r/L) / (r + 1e-6)
