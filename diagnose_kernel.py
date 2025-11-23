# diagnose_kernel.py — Kernel Force Sign & Magnitude Inspector

import json
import numpy as np

from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G
from run_sparc_lite import safe_two_component_disk, choose_box_and_grid, get_U_grid

GAL = "NGC3198"

# Load parameters
with open("results/all_galaxy_params.json", "r") as f:
    params = json.load(f)[GAL]

L = params["L"]
mu = params["mu"]

print(f"\nDIAGNOSTIC FOR {GAL}")
print(f"Parameters: L = {L}, mu = {mu}")

# Small grid for speed
Rmax = 30.0
Lbox, n = choose_box_and_grid(Rmax, L)
print(f"Grid size: Lbox={Lbox}, n={n}")

# Galaxy density
rho, dx = safe_two_component_disk(
    n, Lbox,
    Rd_star=3.19, Mstar=1.91e10, hz_star=0.42,
    Rd_gas=8.0,  Mgas=1.08e10,   hz_gas=0.15
)

# Baryons
phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=np.float32(G))
gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

# Kernel
U = get_U_grid(n, Lbox, L, "ananta-hybrid")
phi_K_raw = conv_fft(rho, U, zero_mode=True)
phi_K = mu * np.float32(G) * phi_K_raw
gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

# Force probe
iz = n // 2
ix = n // 2 + n // 4
iy = n // 2

b = gx_b[ix, iy, iz]
k = gx_K[ix, iy, iz]

print("\n--- FORCE CHECK ---")
print(f"Baryonic acceleration: {b:.3e}")
print(f"Kernel acceleration:   {k:.3e}")

if np.sign(b) == np.sign(k):
    print("\nRESULT: Kernel is ATTRACTIVE (Good)\n")
else:
    print("\nRESULT: Kernel is REPULSIVE (Sign-flip error!)\n")

