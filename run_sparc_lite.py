# run_sparc_lite.py — Harmonized Version (R2-0 + R2-3 Implemented)

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import two_component_disk
from src.kernels import U_plummer, U_exp_core
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- knobs ----
RADIAL_BINS = 30
KERNELS = ("plummer", "exp-core")
# Coarse grid (can be widened as needed)
L_LIST  = [2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0]
MU_LIST = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

# ---- Helper Functions ----

def choose_box_and_grid(R_obs_max, L):
    """
    Chooses an appropriate box size and grid resolution for a given galaxy
    and kernel length scale to avoid numerical artifacts.
    (R2-0 Implementation)
    """
    # Rule: Box must be large enough for the galaxy and the kernel's "tail"
    target_half = max(1.5 * R_obs_max, 6.0 * L, 20.0)
    Lbox = float(target_half)
    
    # Rule: Keep pixel size (dx) reasonable, but cap grid size for speed
    n = int(np.clip(round(2 * Lbox / 0.7), 64, 160))
    
    # Rule: Grid size must be even for easier indexing of the mid-plane
    if n % 2 == 1:
        n += 1
    return Lbox, n

def build_U_grid(n, Lbox, L, kernel):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    if kernel == "plummer":
        U = U_plummer(r, L)
    elif kernel == "exp-core":
        U = U_exp_core(r, L)
    else:
        raise ValueError("kernel must be 'plummer' or 'exp-core'")
    U.flat[0] = 0.0
    return U

U_CACHE = {}
def get_U_grid(n, Lbox, L, kernel):
    """Smarter cache that handles dynamic grid sizes."""
    key = (kernel, float(L), int(n), round(float(Lbox), 3))
    if key not in U_CACHE:
        U_CACHE[key] = build_U_grid(n, Lbox, L, kernel)
    return U_CACHE[key]

def radial_profile_2d(arr2d, dx, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) * dx
    rb = np.linspace(0, (n // 2 - 1) * dx, nbins + 1)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins)
    prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        prof[i] = float(np.mean(arr2d[m])) if np.any(m) else np.nan
    return centers, prof

def read_galaxy_table(path_csv):
    """Corrected version of the CSV reader."""
    out = []
    with open(path_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(x):
                try:
                    return float(x)
                except:
                    return 0.0
            g = {
                "name":    row["name"],
                "Rd_star": num(row["Rd_star_kpc"]),
                "Mstar":   num(row["Mstar_Msun"]),
                "hz_star": num(row["hz_star_kpc"]),
                "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                "Mgas":    num(row.get("Mgas_Msun", "0")),
                "hz_gas":  num(row.get("hz_gas_kpc", "0.3")),
            }
            if g["Rd_gas"] <= 0:
                g["Rd_gas"] = 1.8 * g["Rd_star"]
            out.append(g)
    return out

def try_read_observed_rc(name):
    path = os.path.join("data", "sparc", f"{name}_rc.csv")
    if not os.path.exists(path):
        return None
    R, V = [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            R.append(float(row["R_kpc"]))
            V.append(float(row["V_kms"]))
    return np.array(R), np.array(V)


def predict_rc_for_params(gal, L, mu, kernel):
    """
    Fully updated to return total, baryon, and kernel velocity components.
    (R2-0 Adaptive Box + R2-3 Component Decomposition)
    """
    # 1. Decide box/grid from the galaxy’s observed extent + kernel scale
    obs = try_read_observed_rc(gal["name"])
    R_obs_max = float(np.nanmax(obs[0])) if obs is not None else 15.0
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    # 2. Baryon density on that auto-sized grid
    rho, dx = two_component_disk(
        n, Lbox,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    # 3. Newtonian (Baryon) piece
    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G)
    gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

    # 4. Kernel (Ananta) piece
    U = get_U_grid(n, Lbox, L, kernel)
    phi_K_raw = conv_fft(rho, U, zero_mode=True)
    phi_K = mu * G * phi_K_raw
    gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    # 5. --- R2-3 IMPLEMENTATION: Calculate separate force components ---
    iz = n // 2  # Index for the galactic mid-plane

    # Total force (vector sum)
    gmag_total = np.sqrt((gx_b[:, :, iz] + gx_K[:, :, iz])**2 +
                         (gy_b[:, :, iz] + gy_K[:, :, iz])**2)
    # Baryon-only force
    gmag_baryons = np.sqrt(gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2)
    # Kernel-only force
    gmag_kernel = np.sqrt(gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2)

    # 6. Get radial profiles for each component
    R_centers, g_mean_total = radial_profile_2d(gmag_total, dx, nbins=RADIAL_BINS)
    _, g_mean_baryons = radial_profile_2d(gmag_baryons, dx, nbins=RADIAL_BINS)
    _, g_mean_kernel = radial_profile_2d(gmag_kernel, dx, nbins=RADIAL_BINS)
    
    # 7. Calculate final velocity curves for each component
    v_total = np.sqrt(np.maximum(R_centers * g_mean_total, 0.0))
    v_baryons = np.sqrt(np.maximum(R_centers * g_mean_baryons, 0.0))
    v_kernel = np.sqrt(np.maximum(R_centers * g_mean_kernel, 0.0))
    
    # 8. Return all three curves
    return R_centers, v_total, v_baryons, v_kernel


def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))


def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    table_path = os.path.join("data", "galaxies.csv")
    if not os.path.exists(table_path):
        print("No data/galaxies.csv found.")
        return
    gals = read_galaxy_table(table_path)
    if not gals:
        print("galaxies.csv is empty.")
        return

    print("Starting grid search with adaptive box...")
    best = {"L": None, "mu": None, "mafe": 1e99, "kernel": None}
    
    # --- Grid Search Loop ---
    for kernel in KERNELS:
        for L in L_LIST:
            for mu in MU_LIST:
                scores = []
                for gal in gals:
                    obs = try_read_observed_rc(gal["name"])
                    if obs is None: continue
                    
                    R_obs, V_obs = obs
                    # Unpack the 4 return values (we only need R_pred and V_pred for scoring)
                    R_pred, V_pred, _, _ = predict_rc_for_params(gal, L, mu, kernel)
                    
                    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
                    m = np.isfinite(Vp)
                    if np.any(m):
                        scores.append(mafe(Vp[m], V_obs[m]))

                if scores:
                    med = float(np.median(scores))
                    if med < best["mafe"]:
                        best = {"L": float(L), "mu": float(mu), "mafe": med, "kernel": kernel}
    
    if best["L"] is None:
        best = {"L": 6.0, "mu": 1.0, "mafe": float("nan"), "kernel": "plummer"}
    print("Best (median MAFE):", best)

    # --- Plotting & Export Loop ---
    summary = []
    for gal in gals:
        # Call the function and unpack ALL components for plotting
        R_pred, V_pred, V_b, V_k = predict_rc_for_params(gal, best["L"], best["mu"], best["kernel"])
        obs = try_read_observed_rc(gal["name"])

        out = f"figs/rc_{gal['name']}_{best['kernel']}.png"
        plt.figure(figsize=(5, 4))
        
        # Plot Total Prediction
        plt.plot(R_pred, V_pred, "-", color='orange', lw=2, label=f'H1 Total (L={best["L"]:.1f}, μ={best["mu"]:.2f})')
        
        # Plot Observed Data
        if obs is not None:
            R_obs, V_obs = obs
            plt.plot(R_obs, V_obs, "o", color='white', mec='black', ms=4, mew=0.5, label="Observed RC")
            
            # Calculate score for summary
            Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
            m = np.isfinite(Vp)
            score = mafe(Vp[m], V_obs[m]) if np.any(m) else float("nan")
        else:
            score = float("nan")

        # Plot Diagnostic Overlays (R2-3)
        plt.plot(R_pred, V_b, ':', color='cyan', lw=1.5, label='Baryons-only')
        plt.plot(R_pred, V_k, '--', color='lime', lw=1.5, label='Kernel-only')
        
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — {best['kernel']}")
        
        if obs is not None: # Auto-adjust x-axis to the data range
            plt.xlim(0, np.nanmax(R_obs) * 1.1)
            
        plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        
        summary.append({"name": gal["name"], "mafe": score})
        print("Saved", out)

        # Export Decomposed Data (R2-3)
        np.savetxt(
            f"results/rc_decomp_{gal['name']}_{best['kernel']}.csv",
            np.c_[R_pred, V_b, V_k, V_pred], 
            delimiter=",",
            header="R_kpc,V_baryon_kms,V_kernel_kms,V_total_kms",
            comments=""
        )
        
        if obs is not None:
            np.savetxt(f"results/rc_obs_{gal['name']}.csv", np.c_[R_obs, V_obs], delimiter=",", header="R_kpc,V_kms", comments="")

    with open("results/sparc_lite_best.json", "w") as f:
        json.dump(best, f, indent=2)

    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']}\n")


if __name__ == "__main__":
    main()
