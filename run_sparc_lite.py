# run_sparc_lite.py — Harmonized Version (R2-6 Memory Safe + Gap Filler)

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# Note: We use local safe versions of physics functions to prevent crashes
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

def safe_two_component_disk(n, Lbox, Rd_star, Mstar, hz_star, Rd_gas, Mgas, hz_gas):
    """
    Local, crash-proof version of the galaxy density generator.
    Handles large coordinates safely by clamping cosh().
    """
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x**2 + y**2)
    dx = axis[1] - axis[0]
    
    def get_rho(M, Rd, hz):
        if M <= 0 or Rd <= 0: return np.zeros_like(R)
        hz = max(hz, 1e-3) # Safety floor
        
        # Radial: standard exponential disk
        radial = np.exp(-R / Rd)
        
        # Vertical: sech^2(z/hz) with SAFETY CLAMP
        z_scaled = np.abs(z / hz)
        vertical = np.zeros_like(z)
        # Only calculate where z is reasonable (< 20 scale heights) to avoid overflow
        mask = z_scaled < 20.0 
        vertical[mask] = (1.0 / np.cosh(z_scaled[mask]))**2
        
        # Normalization
        rho0 = M / (4 * np.pi * Rd**2 * hz)
        return rho0 * radial * vertical

    rho_star = get_rho(Mstar, Rd_star, hz_star)
    rho_gas = get_rho(Mgas, Rd_gas, hz_gas)
    
    return rho_star + rho_gas, dx

def choose_box_and_grid(R_obs_max, L):
    # OPTIMIZATION: Reduced padding from 6.0 to 4.0 to save memory
    target_half = max(1.5 * R_obs_max, 4.0 * L, 20.0)
    Lbox = float(target_half)
    
    # SAFETY: Cap grid at 320 to prevent Out-Of-Memory crashes
    # This gives good resolution without killing the server
    n = int(np.clip(round(2 * Lbox / 0.5), 64, 320))
    if n % 2 == 1: n += 1
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
    key = (kernel, float(L), int(n), round(float(Lbox), 3))
    if key not in U_CACHE:
        U_CACHE[key] = build_U_grid(n, Lbox, L, kernel)
    return U_CACHE[key]

def fill_nans(arr):
    """Interpolate small gaps in the profile to prevent broken lines."""
    mask = np.isnan(arr)
    if not np.any(mask): return arr
    if np.all(mask): return arr
    idx = np.where(~mask)[0]
    arr[mask] = np.interp(np.where(mask)[0], idx, arr[idx])
    return arr

def radial_profile_2d(arr2d, dx, max_r, nbins=30):
    """Calculate profile within the galaxy radius and fill gaps."""
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) * dx
    
    # Bin only up to the galaxy edge
    rb = np.linspace(0, max_r * 1.1, nbins + 1)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins)
    prof[:] = np.nan
    
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        if np.any(m):
            prof[i] = float(np.mean(arr2d[m]))
    
    # Fill gaps caused by resolution mismatch
    return centers, fill_nans(prof)

def read_galaxy_table(path_csv):
    out = []
    with open(path_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(x):
                try: return float(x)
                except: return 0.0
            g = {
                "name":    row["name"],
                "Rd_star": num(row["Rd_star_kpc"]),
                "Mstar":   num(row["Mstar_Msun"]),
                "hz_star": num(row["hz_star_kpc"]),
                "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                "Mgas":    num(row.get("Mgas_Msun", "0")),
                "hz_gas":  num(row.get("hz_gas_kpc", "0.3")),
            }
            if g["Rd_gas"] <= 0: g["Rd_gas"] = 1.8 * g["Rd_star"]
            out.append(g)
    return out

def try_read_observed_rc(name):
    path = os.path.join("data", "sparc", f"{name}_rc.csv")
    if not os.path.exists(path): return None
    R, V = [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            R.append(float(row["R_kpc"]))
            V.append(float(row["V_kms"]))
    return np.array(R), np.array(V)

def predict_rc_for_params(gal, L, mu, kernel):
    obs = try_read_observed_rc(gal["name"])
    R_obs_max = float(np.nanmax(obs[0])) if obs is not None else 15.0
    
    # 1. Build Box (Optimized size)
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    # 2. Baryons (Using SAFE local function)
    rho, dx = safe_two_component_disk(
        n, Lbox,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    # 3. Newtonian
    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G)
    gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

    # 4. Kernel
    U = get_U_grid(n, Lbox, L, kernel)
    phi_K_raw = conv_fft(rho, U, zero_mode=True)
    phi_K = mu * G * phi_K_raw
    gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    # 5. Forces
    iz = n // 2 
    gmag_total = np.sqrt((gx_b[:, :, iz] + gx_K[:, :, iz])**2 +
                         (gy_b[:, :, iz] + gy_K[:, :, iz])**2)
    gmag_baryons = np.sqrt(gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2)
    gmag_kernel = np.sqrt(gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2)

    # 6. Profiles (Zoomed + Filled)
    R_centers, g_mean_total = radial_profile_2d(gmag_total, dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_baryons = radial_profile_2d(gmag_baryons, dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_kernel = radial_profile_2d(gmag_kernel, dx, R_obs_max, nbins=RADIAL_BINS)
    
    # 7. Velocities
    v_total = np.sqrt(np.maximum(R_centers * g_mean_total, 0.0))
    v_baryons = np.sqrt(np.maximum(R_centers * g_mean_baryons, 0.0))
    v_kernel = np.sqrt(np.maximum(R_centers * g_mean_kernel, 0.0))
    
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

    print("Starting grid search with MEMORY-SAFE settings...")
    best = {"L": None, "mu": None, "mafe": 1e99, "kernel": None}
    
    for kernel in KERNELS:
        for L in L_LIST:
            for mu in MU_LIST:
                scores = []
                for gal in gals:
                    obs = try_read_observed_rc(gal["name"])
                    if obs is None: continue
                    
                    R_obs, V_obs = obs
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

    summary = []
    for gal in gals:
        R_pred, V_pred, V_b, V_k = predict_rc_for_params(gal, best["L"], best["mu"], best["kernel"])
        obs = try_read_observed_rc(gal["name"])

        out = f"figs/rc_{gal['name']}_{best['kernel']}.png"
        plt.figure(figsize=(5, 4))
        
        plt.plot(R_pred, V_pred, "-", color='orange', lw=2, label=f'H1 Total (L={best["L"]:.1f}, μ={best["mu"]:.2f})')
        
        if obs is not None:
            R_obs, V_obs = obs
            plt.plot(R_obs, V_obs, "o", color='white', mec='black', ms=4, mew=0.5, label="Observed RC")
            Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
            m = np.isfinite(Vp)
            score = mafe(Vp[m], V_obs[m]) if np.any(m) else float("nan")
        else:
            score = float("nan")

        plt.plot(R_pred, V_b, ':', color='cyan', lw=1.5, label='Baryons-only')
        plt.plot(R_pred, V_k, '--', color='lime', lw=1.5, label='Kernel-only')
        
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — {best['kernel']}")
        
        if obs is not None:
            plt.xlim(0, np.nanmax(R_obs) * 1.1)
            
        plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        
        summary.append({"name": gal["name"], "mafe": score})
        print("Saved", out)

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
