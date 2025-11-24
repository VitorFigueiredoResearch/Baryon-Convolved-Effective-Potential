# src/newtonian.py
# Newtonian potential from 3D density via Poisson in Fourier space.

import numpy as np

G = 4.30091e-6  # kpc (km/s)^2 / Msun

def phi_newtonian_from_rho(rho, Lbox, Gval=G):
    """
    Solve ∇^2 φ_b = 4π G ρ  via FFT:
      φ_k = -4π G ρ_k / k^2   (set k=0 mode to 0)
    Returns φ_b in km^2/s^2 (since G carries those units).
    """
    n = rho.shape[0]
    dx = (2.0 * Lbox) / n
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz

    rho_k = np.fft.fftn(rho, norm=None)
    phi_k = np.zeros_like(rho_k, dtype=complex)
    mask = (k2 != 0.0)
    phi_k[mask] = +4.0 * np.pi * Gval * rho_k[mask] / k2[mask]
    phi_k[~mask] = 0.0
    phi = np.fft.ifftn(phi_k).real
    return phi
