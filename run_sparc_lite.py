# run_sparc_lite.py — Harmonized Version with R2-0 Adaptive Box Implemented

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# These imports will be used in a later refinement (R2-1)
# from dataclasses import dataclass
# import yaml, argparse

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

## R2-0 IMPLEMENTATION: This is the new "smart" box selector
def choose_box_and_grid(R_obs_max, L):
    """
    Chooses an appropriate box size and grid resolution for a given galaxy
    and kernel length scale to avoid numerical artifacts.
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
    """Fully updated prediction function with auto-sizing grid."""
    ## R2-0 IMPLEMENTATION: The adaptive grid is now called here
    obs = try_read_observed_rc(gal["name"])
    R_obs_max = float(np.nanmax(obs[0])) if obs is not None else 15.0
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    rho, dx = two_component_disk(
        n, Lbox,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G)
    gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

    U = get_U_grid(n, Lbox, L, kernel)
    phi_K_raw = conv_fft(rho, U, zero_mode=True)
    phi_K = mu * G * phi_K_raw
    gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    iz = n // 2
    gmag = np.sqrt((gx_b[:, :, iz] + gx_K[:, :, iz])**2 +
                   (gy_b[:, :, iz] + gy_K[:, :, iz])**2)
    R_centers, g_mean = radial_profile_2d(gmag, dx, nbins=RADIAL_BINS)
    vpred = np.sqrt(np.maximum(R_centers * g_mean, 0.0))
    return R_centers, vpred


def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))


def main():
    # ... (The main function remains largely the same, but the calls inside
    #      will now use the updated predict_rc_for_params function)
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
    for kernel in KERNELS:
        for L in L_LIST:
            for mu in MU_LIST:
                scores = []
                for gal in gals:
                    obs = try_read_observed_rc(gal["name"])
                    if obs is None: continue
                    
                    R_obs, V_obs = obs
                    # This call now uses the fully updated, adaptive function
                    R_pred, V_pred = predict_rc_for_params(gal, L, mu, kernel)
                    
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

    summary = []
    for gal in gals:
        R_pred, V_pred = predict_rc_for_params(gal, best["L"], best["mu"], best["kernel"])
        obs = try_read_observed_rc(gal["name"])

        out = f"figs/rc_{gal['name']}_{best['kernel']}.png"
        plt.figure(figsize=(5, 4))
        plt.plot(R_pred, V_pred, "-", lw=2, label=f'H1 pred (L={best["L"]:.1f} kpc, μ={best["mu"]:.2f})')
        if obs is not None:
            R_obs, V_obs = obs
            plt.plot(R_obs, V_obs, "o", ms=3, label="observed RC")
            Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
            m = np.isfinite(Vp)
            score = mafe(Vp[m], V_obs[m]) if np.any(m) else float("nan")
        else:
            score = float("nan")
        
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — {best['kernel']}")
        plt.legend()
        if obs is not None: # Auto-adjust x-axis to the data range
            plt.xlim(0, np.nanmax(R_obs) * 1.1)
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        summary.append({"name": gal["name"], "mafe": score})
        print("Saved", out)

        np.savetxt(f"results/rc_pred_{gal['name']}_{best['kernel']}.csv", np.c_[R_pred, V_pred], delimiter=",", header="R_kpc,Vpred_kms", comments="")
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
