import numpy as np

def set_k0_to_zero(phi_k):
    phi_k = phi_k.copy()
    phi_k.flat[0] = 0.0
    return phi_k

def grad_from_phi_k(phi_k, kx, ky, kz):
    i = 1j
    gx_k = i * kx * phi_k
    gy_k = i * ky * phi_k
    gz_k = i * kz * phi_k
    return gx_k, gy_k, gz_k
