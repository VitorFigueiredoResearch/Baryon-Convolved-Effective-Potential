from src.kernels import U_plummer, U_exp_core, U_ananta_hybrid

import numpy as np

def U_plummer(r, L):
    """
    Standard Plummer kernel (The 'Old' Engine).
    Decays as 1/r^3. Good for cores, bad for outer halos.
    """
    L = float(L)
    # Avoid division by zero
    r_safe = np.maximum(r, 1e-6)
    return (3.0 / (4.0 * np.pi * L**3)) * (1.0 + (r_safe/L)**2)**(-2.5)

def U_ananta_hybrid(r, L):
    """
    The 'Ananta' Kernel (Option C).
    Hybrid: Short-range Gaussian + Long-range Lorentzian Tail.
    
    Physics:
    - Core: Represents local active time/coherence (Strong).
    - Tail: Represents the persistent Recorded Past (Long-range Viscosity).
    """
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    
    # 1. The Core (Gaussian) - Holds the center
    # Scaled to integrate to ~0.5 mass fraction
    core = (1.0 / (np.pi * L**2)**1.5) * np.exp(-(r_safe**2) / (L**2))
    
    # 2. The Tail (Lorentzian-like) - Holds the edge
    # Decays slower than Plummer, mimicking a Dark Matter Halo profile.
    # Form: 1 / (1 + (r/L)^2)^1.5  (similar to Burkert/Isothermal)
    tail_norm = 1.0 / (4.0 * np.pi * L**3) # Approximate norm
    tail = tail_norm * (1.0 + (r_safe/L)**2)**(-1.0) 
    
    # Combine them (Equal weight for now, tunable via mu later)
    return 0.5 * core + 0.5 * tail

def U_exp_core(r, L):
    """
    Exponential kernel.
    Decays too fast for large galaxies. Kept for legacy comparison.
    """
    L = float(L)
    norm = 1.0 / (8.0 * np.pi * L**3)
    return norm * np.exp(-r/L)
