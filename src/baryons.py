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
def two_component_disk(n, Lbox,
                       Rd_star, Mstar, hz_star,
                       Rd_gas=0.0, Mgas=0.0, hz_gas=0.3):
    """Return ρ_b = ρ_star (+ ρ_gas if provided). Masses in Msun, lengths in kpc."""
    rho_star, dx = make_exponential_disk(n=n, Lbox=Lbox, Rd=Rd_star, Md=Mstar, hz=hz_star)
    rho_tot = rho_star.copy()
    if (Mgas > 0.0) and (Rd_gas > 0.0):
        rho_gas, _ = make_exponential_disk(n=n, Lbox=Lbox, Rd=Rd_gas, Md=Mgas, hz=hz_gas)
        rho_tot += rho_gas
    return rho_tot, dx
