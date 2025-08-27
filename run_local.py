# run_local.py — first pictures from either kernel

import os
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import make_exponential_disk
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.kernels import U_plummer, U_exp_core

# choose: "plummer" or "exp-core"
KERNEL = "plummer"   # change to "exp-core" to compare
L = 6.0              # kpc

# 1) mass map
n = 64
Lbox = 20.0
rho, dx = make_exponential_disk(n=n, Lbox=Lbox, Rd=3.0, Md=5e10, hz=0.3)

# 2) build U(r;L) on the same grid
axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
r = np.sqrt(x*x + y*y + z*z)

if KERNEL == "plummer":
    U = U_plummer(r, L)
elif KERNEL == "exp-core":
    U = U_exp_core(r, L)
else:
    raise ValueError("KERNEL must be 'plummer' or 'exp-core'")
U.flat[0] = 0.0  # safe at the origin

# 3) convolution and gradient
phiK = conv_fft(rho, U, zero_mode=True)
gx, gy, gz = gradient_from_phi(phiK, Lbox)

# 4) sample a toy “rotation curve” on y=z=0
ix0 = iy0 = iz0 = n // 2
indices = np.arange(ix0 + 1, n - 2)
R = (indices - ix0) * dx
gR = np.sqrt(gx[indices, iy0, iz0]**2 + gy[indices, iy0, iz0]**2)
v_proxy = np.sqrt(np.maximum(R * gR, 0.0))

# 5) save figure
os.makedirs("figs", exist_ok=True)
out = f"figs/rc_toy_{KERNEL}.png"
plt.figure(figsize=(5, 4))
plt.plot(R, v_proxy, "-o", ms=3)
plt.xlabel("R [kpc]"); plt.ylabel("toy speed (arb. units)")
plt.title(f"Toy curve from {KERNEL} kernel (L = {L:.1f} kpc)")
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Done. Figure saved to {out}")
