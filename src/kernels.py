import numpy as np

def K_plummer(r, L):
    return 1.0 / (4.0 * np.pi * (r**2 + L**2))

def K_exp_core(r, L):
    # Cored exponential: e^{-r/L} / [4π (r^2 + L^2)]
    return np.exp(-r / L) / (4.0 * np.pi * (r**2 + L**2))

def U_hat_isotropic(k, L, kernel_type="plummer"):
    """Return \hat{U}(k) for isotropic kernel; numerical forms acceptable.
    For Plummer: U(r) = (4πL)^{-1} arctan(r/L) ⇒ \hat{U}(k) ~ f(kL).
    Implemented as a placeholder (user to supply calibrated FFT form).
    """
    # Placeholder smooth low-pass form; replace with calibrated transform.
    x = k * L
    if kernel_type == "plummer":
        return 1.0 / (1.0 + x**2)
    elif kernel_type == "exp-core":
        return np.exp(-x) / (1.0 + x**2)
    else:
        raise ValueError("Unknown kernel_type")
