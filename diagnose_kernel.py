# diagnose_kernel.py — Kernel Force Sign & Magnitude Inspector

import json
import numpy as np

from src.kernels import U_ananta_hybrid
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G
from run_sparc_lite import safe_two_component_disk, choose_box_and_grid, get_U_grid

GAL = "NGC3198"

# 1. Load parameters
with open("results/all_galaxy_params.json", "r") as f:
    params = json.load(f)[GAL]

L = params["L"]
mu = params["mu"]

print(f"DIAGNOSTIC FOR {GAL}")
print(f"Parameters: L = {L}, mu = {mu}")

# 2. Make small test grid
Rmax = 30.0
Lbox, n = choose_box_and_grid(Rmax, L)
print(f"Grid: Lbox={Lbox}, n={n}")

# 3. Build galaxy density
rho, dx = safe_two_component_disk(
    n, Lbox,
    Rd_star=3.19, Mstar=1.91e10, hz_star=0.42,
    Rd_gas=8.0, Mgas=1.08e10, hz_gas=0.15
)

# 4. Baryon field
phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=np.float32(G))
gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

# 5. Kernel field
U = get_U_grid(n, Lbox, L, "ananta-hybrid")
phi_K_raw = conv_fft(rho, U, zero_mode=True)
phi_K = mu * np.float32(G) * phi_K_raw
gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

# 6. Evaluate forces at a representative point
iz = n // 2
ix = n // 2 + n // 4  # ~25% outwards on x-axis
iy = n // 2

b = gx_b[ix, iy, iz]
k = gx_K[ix, iy, iz]

print("\n--- FORCE CHECK ---")
print(f"Baryon pull: {b:.3e}")
print(f"Kernel pull: {k:.3e}")

if np.sign(b) == np.sign(k):
    print("RESULT: Kernel is ATTRACTIVE (good)")
else:
    print("RESULT: Kernel is REPULSIVE (sign error)")
