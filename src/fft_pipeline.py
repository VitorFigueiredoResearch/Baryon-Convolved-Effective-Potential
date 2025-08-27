# src/fft_pipeline.py
# Convolution by FFT and gradient via k-space. Simple, fast, and matches H1.

import numpy as np

def set_k0_to_zero(phi_k):
    """
    Zero the DC (k=0) mode to remove any box-constant offset in the potential.
    Safe because ∇phi is unchanged by adding a constant.
    """
    phi_k = phi_k.copy()
    phi_k.flat[0] = 0.0
    return phi_k

def conv_fft(rho, U, zero_mode=True):
    """
    Convolve density 'rho' with scalar kernel 'U' using FFTs.
    Both arrays must be the same shape (n x n x n). Returns phi = rho * U.

    Tip: rho is your 3D baryon map, U is built from L (the kernel length).
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
    Compute ∇phi using Fourier derivatives.

    Args:
      phi  : 3D array (n x n x n) potential
      Lbox : half-size of the box in kpc; grid spans [-Lbox, +Lbox]

    Returns:
      gx, gy, gz : 3D arrays, components of the gradient (same shape as phi)
    """
    n = phi.shape[0]
    # Physical grid spacing (kpc)
    dx = (2.0 * Lbox) / n
    # Wave numbers in physical units (1/kpc)
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')

    phi_k = np.fft.fftn(phi, norm=None)
    # Gradient in Fourier space: ∇phi  ->  i * k * phi_k
    i = 1j
    gx_k = i * kx * phi_k
    gy_k = i * ky * phi_k
    gz_k = i * kz * phi_k

    gx = np.fft.ifftn(gx_k).real
    gy = np.fft.ifftn(gy_k).real
    gz = np.fft.ifftn(gz_k).real
    return gx, gy, gz
