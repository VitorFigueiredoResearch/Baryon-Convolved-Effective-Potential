import numpy as np

def U_plummer(r, L):
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    return (3.0 / (4.0 * np.pi * L**3)) * (1.0 + (r_safe/L)**2)**(-2.5)

def U_ananta_hybrid(r, L):
    """
    The 'Ananta' Kernel (Original Hybrid).
    Core: Gaussian (Coherence). Tail: Lorentzian (Memory).
    This is scientifically robust and normalizable.
    """
    L = float(L)
    r_safe = np.maximum(r, 1e-6)
    
    # Core (Gaussian)
    core = (1.0 / (np.pi * L**2)**1.5) * np.exp(-(r_safe**2) / (L**2))
    
    # Tail (Lorentzian-like)
    tail_norm = 1.0 / (4.0 * np.pi * L**3)
    tail = tail_norm * (1.0 + (r_safe/L)**2)**(-1.0) 
    
    return 0.5 * core + 0.5 * tail

def U_exp_core(r, L):
    L = float(L)
    norm = 1.0 / (8.0 * np.pi * L**3)
    return norm * np.exp(-r/L)
