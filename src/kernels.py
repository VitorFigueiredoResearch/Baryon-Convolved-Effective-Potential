import numpy as np

def U_plummer(r, L):
    """
    Standard Plummer Potential (Newtonian-like).
    Units: 1/r
    """
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    return 1.0 / np.sqrt(r_safe**2 + L**2)

def U_ananta_hybrid(r, L):
    """
    R3-6: The 'Ananta' Logarithmic Potential Kernel.
    
    Physics:
    - A Logarithmic Potential produces a Force ~ 1/r.
    - Force ~ 1/r leads to a Flat Rotation Curve (V^2 ~ constant).
    - This mimics the '2D spread' of the Recorded Past.
    
    Normalization:
    - To ensure the force has the right magnitude (comparable to Newtonian),
      we scale it by a factor S = 100.0 / L.
    - This ensures that at r=L, the force is significant.
    """
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    
    # Scaling Factor: Boosts the signal to match galactic scales
    # Without this, the log function is too 'flat' relative to G.
    scale_factor = 100.0 / L
    
    # Logarithmic Potential
    # 0.5 * ln(1 + (r/L)^2)
    return scale_factor * 0.5 * np.log(1.0 + (r_safe/L)**2)

def U_exp_core(r, L):
    """
    Exponential Potential (Yukawa-like).
    """
    L = float(L)
    # Approximate potential form of exponential density
    return np.exp(-r/L) / (r + 1e-6)
