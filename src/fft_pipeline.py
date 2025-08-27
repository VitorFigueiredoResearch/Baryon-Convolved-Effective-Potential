# src/fft_pipeline.py
# Convolution by FFT and gradient/Laplacian via k-space.

import numpy as np

def set_k0_to_zero(phi_k):
    """Zero the DC (k=0) mode; safe for ∇phi."""
    phi_k = phi_k.copy()
    phi_k.flat[0] = 0.0
    return phi_k

def conv_fft(rho, U, zero_mode=True):
    """
    Convolve density 'rho' with scalar kernel 'U' using FFTs.
    Both arrays must have the same shape (n x n x n). Returns phi = rho * U.
    """
    rho_k = np.fft.fftn(rho, norm=None)
    U_k   = np.fft.fftn(U,   norm=None)
    phi_k = rho_k * U_k
    if zero_mode:
        phi_k = set_k0_to_zero(phi_k)
    phi = np.fft.ifftn(phi_k).real
    return phi

def gradient_from_phi(phi, Lbox):
    """
    Compute ∇phi via Fourier derivatives.
    phi : 3D array; Lbox : half-size of box in kpc (grid spans [-Lbox, +Lbox]).
    """
    n = phi.shape[0]
    dx = (2.0 * Lbox) / n
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')

    phi_k = np.fft.fftn(phi, norm=None)
    i = 1j
    gx = np.fft.ifftn(i * kx * phi_k).real
    gy = np.fft.ifftn(i * ky * phi_k).real
    gz = np.fft.ifftn(i * kz * phi_k).real
    return gx, gy, gz

def laplacian_from_phi(phi, Lbox):
    """Return ∇²phi via Fourier: (∇²)↔(-k^2)."""
    n = phi.shape[0]
    dx = (2.0 * Lbox) / n
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz
    phi_k = np.fft.fftn(phi, norm=None)
    lap_k = -k2 * phi_k
    lap_k.flat[0] = 0.0
    lap = np.fft.ifftn(lap_k).real
    return lap
