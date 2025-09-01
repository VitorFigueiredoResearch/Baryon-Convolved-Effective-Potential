# run_local.py — RC proxy + Σ(R) + ΔΣ(R) for both kernels

import os
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import make_exponential_disk
from src.fft_pipeline import conv_fft, gradient_from_phi, laplacian_from_phi
from src.kernels import U_plummer, U_exp_core
from src.newtonian import phi_newtonian_from_rho, G
from src.fft_pipeline import laplacian_from_phi

# global knobs
n = 64
Lbox = 20.0  # kpc
L = 6.0      # kpc (kernel length)

def build_U(r, kernel):
    if kernel == "plummer":
        return U_plummer(r, L)
    elif kernel == "exp-core":
        return U_exp_core(r, L)
    else:
        raise ValueError("kernel must be 'plummer' or 'exp-core'")

def radial_profile(arr2d, dx, nbins=30):
    """Azimuthal mean vs radius for a 2D array sampled on square pixels of size dx."""
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5)**2 + (yy - cy + 0.5)**2) * dx
    rbins = np.linspace(0, (n//2 - 1)*dx, nbins+1)
    centers = 0.5 * (rbins[1:] + rbins[:-1])
    prof = []
    for r0, r1 in zip(rbins[:-1], rbins[1:]):
        m = (R >= r0) & (R < r1)
        prof.append(float(np.mean(arr2d[m])) if np.any(m) else np.nan)
    return centers, np.array(prof), rbins, R

def run_for_kernel(kernel):
    # 1) mass map
    rho, dx = make_exponential_disk(n=n, Lbox=Lbox, Rd=3.0, Md=5e10, hz=0.3)

    # 2) kernel grid
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x*x + y*y + z*z)
    U = build_U(r, kernel)
    U.flat[0] = 0.0

    # 3) φ_K and ∇φ_K
    phiK = conv_fft(rho, U, zero_mode=True)
    gx, gy, gz = gradient_from_phi(phiK, Lbox)

   # --- RC proxy (mid-plane, ring-averaged) ---
    iz0 = n // 2
    gR2d = np.sqrt(gx[:, :, iz0]**2 + gy[:, :, iz0]**2)  # force magnitude in plane
    centers_RC, gmean, _, _ = radial_profile(gR2d, dx, nbins=30)  # azimuthal mean per ring
    v_proxy = np.sqrt(np.maximum(centers_RC * gmean, 0.0))

    os.makedirs("figs", exist_ok=True)
    out_rc = f"figs/rc_toy_{kernel}.png"
    plt.figure(figsize=(5, 4))
    plt.plot(centers_RC, v_proxy, "-o", ms=3)
    plt.xlabel("R [kpc]"); plt.ylabel("toy speed (arb. units)")
    plt.title(f"Toy curve from {kernel} kernel (L = {L:.1f} kpc)")
    plt.tight_layout(); plt.savefig(out_rc, dpi=150); plt.close()


    # --- Σ(R) from the same field (toy) ---
    rho_eff = laplacian_from_phi(phiK, Lbox)      # ∝ ∇²φ (arb. units)
    Sigma = np.sum(rho_eff, axis=2) * dx          # project along z
    centers, Smean, rbins, R2D = radial_profile(Sigma, dx, nbins=30)

    out_sig = f"figs/sigma_toy_{kernel}.png"
    plt.figure(figsize=(5,4))
    plt.plot(centers, Smean, "-o", ms=3)
    plt.xlabel("R [kpc]"); plt.ylabel(r"$\Delta\Sigma$ [$M_\odot \, \mathrm{pc}^{-2}$]")
    plt.title(f"Projected Σ from {kernel} kernel")
    plt.tight_layout(); plt.savefig(out_sig, dpi=150); plt.close()

   # --- Σ(R) and ΔΣ(R) in physical units ---
    # Build Newtonian baryon potential (physical units via G), then add kernel part.
    phi_b  = phi_newtonian_from_rho(rho, Lbox, Gval=G)
    phi_tot = phi_b + phiK  # IMPORTANT: lensing uses total potential

    # 3D Laplacian → total volume density, then project to Σ
    lap_tot = laplacian_from_phi(phi_tot, Lbox)      # ∇²φ_tot
    rho_tot = lap_tot / (4.0 * np.pi * G)            # ρ_tot = (∇²φ_tot)/(4πG)
    Sigma_kpc2 = np.sum(rho_tot, axis=2) * dx         # Project along z. Units: Msun/kpc^2

    # <<< THIS IS THE PATCH >>>
    # Convert Σ from Msun/kpc^2 to Msun/pc^2 for standard plotting
    KPC2_PER_PC2 = 1.0e6
    Sigma_phys   = Sigma_kpc2 / KPC2_PER_PC2

    # Now, calculate all radial profiles using the physically-scaled Sigma_phys
    centers, Smean, rbins, R2D = radial_profile(Sigma_phys, dx, nbins=30)

    # ΔΣ in Msun/pc^2
    cum_mean = []
    for r1 in rbins[1:]:
        m = (R2D < r1)
        cum_mean.append(float(np.mean(Sigma_phys[m])) if np.any(m) else np.nan)
    cum_mean   = np.array(cum_mean)
    DeltaSigma = cum_mean - Smean
    # <<< END OF PATCH >>>

    # Non-negativity diagnostic (should be ≥0 within toy noise)
    frac_neg = float(np.mean(Sigma_phys < 0.0))
    print(f"[diag] Σ_tot: min={Sigma_phys.min():.3e}, max={Sigma_phys.max():.3e}, frac<0={frac_neg:.3%}")

    out_sig  = f"figs/sigma_toy_{kernel}.png"
    out_dsig = f"figs/delta_sigma_toy_{kernel}.png"

    # Update plot labels to reflect new physical units
    plt.figure(figsize=(5,4))
    plt.plot(centers, Smean, "-o", ms=3)
    plt.xlabel("R [kpc]"); plt.ylabel(r"$\Sigma$ [$M_\odot \, \mathrm{pc}^{-2}$]")
    plt.title(f"Projected Surface Density (total) from {kernel} kernel")
    plt.tight_layout(); plt.savefig(out_sig, dpi=150); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(centers, DeltaSigma, "-o", ms=3)
    plt.xlabel("R [kpc]"); plt.ylabel(r"$\Delta\Sigma$ [$M_\odot \, \mathrm{pc}^{-2}$]")
    plt.title(f"Differential Surface Density (total) from {kernel} kernel")
    plt.tight_layout(); plt.savefig(out_dsig, dpi=150); plt.close()

    print(f"Saved: {out_rc}, {out_sig}, {out_dsig}")
