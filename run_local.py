# run_local.py
# First picture: a toy curve using your kernels + FFT pipeline.

import os
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import make_exponential_disk
from src.fft_pipeline import conv_fft, gradient_from_phi

# 1) Toy galaxy mass map
n = 64
Lbox = 20.0  # kpc
rho, dx = make_exponential_disk(n=n, Lbox=Lbox, Rd=3.0, Md=5e10, hz=0.3)

# 2) Build a Plummer *potential* kernel U(r; L) (closed-form)
L = 6.0  # kpc — try changing this later
axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
r = np.sqrt(x*x + y*y + z*z)
U = (1.0 / (4.0 * np.pi * max(L, 1e-6))) * np.arctan(r / max(L, 1e-6))
U.flat[0] = 0.0  # safe at origin

# 3) Convolution → potential φ_K, then gradient ∇φ_K
phiK = conv_fft(rho, U, zero_mode=True)
gx, gy, gz = gradient_from_phi(phiK, Lbox)

# 4) Sample a toy “rotation curve” along the mid-plane (y=0, z=0)
ix0 = n // 2
iy0 = n // 2
iz0 = n // 2
indices = np.arange(ix0 + 1, n - 2)
R = (indices - ix0) * dx  # kpc
gR = np.sqrt(gx[indices, iy0, iz0]**2 + gy[indices, iy0, iz0]**2)

# Quick proxy for speed: sqrt(R * |g|) — arbitrary units (just to see a curve)
v_proxy = np.sqrt(np.maximum(R * gR, 0.0))

# 5) Save the figure
os.makedirs("figs", exist_ok=True)
plt.figure(figsize=(5, 4))
plt.plot(R, v_proxy, "-o", ms=3)
plt.xlabel("R [kpc]")
plt.ylabel("toy speed (arb. units)")
plt.title(f"Toy curve from Plummer kernel (L = {L:.1f} kpc)")
plt.tight_layout()
plt.savefig("figs/rc_toy.png", dpi=150)
plt.close()

print("Done. Figure saved to figs/rc_toy.png")
