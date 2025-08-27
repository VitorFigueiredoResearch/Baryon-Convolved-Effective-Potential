import numpy as np

def make_exponential_disk(n=64, Lbox=20.0, Rd=3.0, Md=5e10, hz=0.3):
    """
    Build a 3D exponential disk on an n×n×n cube spanning [-Lbox, +Lbox] kpc.
    Returns:
      rho : 3D array (density, toy-normalized to total Md)
      dx  : grid cell size (kpc)
    """
    # grid
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x*x + y*y)

    # profiles (radial exp, vertical sech^2)
    Sigma = np.exp(-R / max(Rd, 1e-6))
    sech2 = 1.0 / np.cosh(z / max(hz, 1e-6))**2
    rho = Sigma * sech2

    # normalize total mass to Md (toy)
    dx = (2 * Lbox) / n
    M_now = np.sum(rho) * dx**3
    if M_now > 0:
        rho *= Md / M_now
    return rho, dx
