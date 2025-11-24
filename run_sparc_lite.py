# run_sparc_lite.py — R3-7 "CLEAN / SINGLE GALAXY / DIAGNOSTIC" (NGC3198)
# Single-galaxy debug mode, keeps diagnostics inline and applies the polarity
# reversal (phi_K = - mu * G * phi_K_raw) so the kernel contribution is easily
# visible while we debug normalization/shape. Safe guards for empty input files
# and sanitized filenames included.

import os
import csv
import json
import gc
import urllib.request
import zipfile
import re
import numpy as np
import matplotlib.pyplot as plt

# Local physics modules (must be importable from project root)
from src.kernels import U_plummer, U_exp_core, U_ananta_hybrid
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- SETTINGS ----
RADIAL_BINS = 30

# We'll only test NGC3198 to save time
TARGET_GALAXY = "NGC3198"

# Kernel / grid parameter lists used during the grid search
KERNELS = ("ananta-hybrid",)
L_LIST  = [10.0, 30.0, 50.0, 80.0, 120.0, 200.0]
MU_LIST = [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]

# If CSV or SPARC data not found, fallback to this minimal entry
NIGHTMARE_FLEET = [
    {"name": "NGC3198", "Rd_star": 3.19, "Mstar": 1.91e10, "hz_star": 0.42,
     "Rd_gas": 8.0, "Mgas": 1.08e10, "hz_gas": 0.15},
]

# ---- UTILITIES ----

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name)

def download_and_extract_data():
    data_dir = "data/sparc"
    os.makedirs(data_dir, exist_ok=True)
    url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
    zip_path = os.path.join("data", "Rotmod_LTG.zip")
    existing_files = [f for f in os.listdir(data_dir) if f.endswith("_rotmod.dat")]
    if len(existing_files) < 5:
        print(">>> AUTOMATION: SPARC data missing. Initiating Download Sequence...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"!!! DOWNLOAD ERROR: {e}")

# ---- PHYSICS ENGINE (light but consistent units) ----

def safe_two_component_disk(n, Lbox, Rd_star, Mstar, hz_star, Rd_gas, Mgas, hz_gas):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x**2 + y**2)
    dx = axis[1] - axis[0]

    def get_rho(M, Rd, hz):
        if M <= 0 or Rd <= 0:
            return np.zeros_like(R, dtype=np.float32)
        hz = max(hz, 1e-3)
        radial = np.exp(-R / Rd)
        z_scaled = np.abs(z / hz)
        vertical = np.zeros_like(z, dtype=np.float32)
        mask = z_scaled < 20.0
        vertical[mask] = (1.0 / np.cosh(z_scaled[mask]))**2
        rho0 = M / (4.0 * np.pi * Rd**2 * hz)  # Msun / kpc^3
        return (rho0 * radial * vertical).astype(np.float32)

    rho_star = get_rho(Mstar, Rd_star, hz_star)
    rho_gas = get_rho(Mgas, Rd_gas, hz_gas)
    return rho_star + rho_gas, dx

def choose_box_and_grid(R_obs_max, L):
    target_half = max(1.5 * R_obs_max, 4.0 * L, 20.0)
    Lbox = float(target_half)
    # dx target ~0.5 kpc, n clipped to [64, 320]
    n = int(np.clip(round(2.0 * Lbox / 0.5), 64, 320))
    if n % 2 == 1:
        n += 1
    return Lbox, n

def build_U_grid(n, Lbox, L, kernel):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    if kernel == "plummer":
        U = U_plummer(r, L)
        # give finite center for density-like kernels (if used)
        U.flat[0] = max(U.flat[0], 1.0 / max(1e-3, float(L)))
    elif kernel == "exp-core":
        U = U_exp_core(r, L)
        U.flat[0] = max(U.flat[0], 1.0 / max(1e-3, float(L) * 2.718))
    elif kernel == "ananta-hybrid":
        # Use latest hybrid/potential kernel from src/kernels
        U = U_ananta_hybrid(r, L)
        # DON'T forcibly zero center for potential-like kernels
    else:
        raise ValueError("kernel error")
    return U.astype(np.float32)

# simple cache to avoid rebuilding identical grids multiple times
U_CACHE = {}
def get_U_grid(n, Lbox, L, kernel):
    key = (kernel, float(L), int(n), round(float(Lbox), 2))
    if key not in U_CACHE:
        U_CACHE[key] = build_U_grid(n, Lbox, L, kernel)
    return U_CACHE[key]

def fill_nans(arr):
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    if np.all(mask):
        return arr
    idx = np.where(~mask)[0]
    arr[mask] = np.interp(np.where(mask)[0], idx, arr[idx])
    return arr

def radial_profile_2d(arr2d, dx, max_r, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt(((xx - cx + 0.5) * dx)**2 + ((yy - cy + 0.5) * dx)**2).astype(np.float32)
    rb = np.linspace(0.0, max_r * 1.1, nbins + 1).astype(np.float32)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins, dtype=np.float32)
    prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        if np.any(m):
            prof[i] = np.mean(arr2d[m])
    return centers, fill_nans(prof)

# ---- DATA IO ----

def read_galaxy_table(path_csv):
    out = []
    if os.path.exists(path_csv):
        try:
            with open(path_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # FAST MODE: only keep the target galaxy
                    name = row.get("name", "").strip()
                    if name != TARGET_GALAXY:
                        continue
                    def num(x):
                        try:
                            return float(x)
                        except Exception:
                            return 0.0
                    g = {
                        "name": name,
                        "Rd_star": num(row.get("Rd_star_kpc", "0")),
                        "Mstar": num(row.get("Mstar_Msun", "0")),
                        "hz_star": num(row.get("hz_star_kpc", "0.3")),
                        "Rd_gas": num(row.get("Rd_gas_kpc", "0")),
                        "Mgas": num(row.get("Mgas_Msun", "0")),
                        "hz_gas": num(row.get("hz_gas_kpc", "0.15")),
                    }
                    if g["Rd_gas"] <= 0:
                        g["Rd_gas"] = 1.8 * g["Rd_star"]
                    out.append(g)
        except Exception as e:
            print(f"Note: Error reading CSV ({e})")

    if not out:
        print(">>> Using hardcoded fallback (NGC3198)")
        return NIGHTMARE_FLEET
    return out

def try_read_observed_rc(name):
    base_dirs = ["data/sparc", "data/sparc/Rotmod_LTG"]
    file_to_read = None
    is_dat = False
    for d in base_dirs:
        path_dat = os.path.join(d, f"{name}_rotmod.dat")
        path_csv = os.path.join(d, f"{name}_rc.csv")
        if os.path.exists(path_dat):
            file_to_read, is_dat = path_dat, True
            break
        elif os.path.exists(path_csv):
            file_to_read, is_dat = path_csv, False
            break
    if file_to_read is None:
        return None

    R = []
    V = []
    try:
        with open(file_to_read, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Rad"):
                    continue
                if is_dat:
                    parts = line.split()
                else:
                    parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        R.append(float(parts[0]))
                        V.append(float(parts[1]))
                    except ValueError:
                        continue
        if len(R) == 0:
            return None
        return np.array(R, dtype=np.float32), np.array(V, dtype=np.float32)
    except Exception:
        return None

# ---- CORE PIPELINE ----

def predict_rc_for_params(gal, L, mu, kernel):
    obs = try_read_observed_rc(gal["name"])
    # guard against missing/empty observed data
    if obs is None or obs[0].size == 0:
        return None

    R_obs_max = float(np.nanmax(obs[0]))
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    rho, dx = safe_two_component_disk(
        n, Lbox,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    G32 = np.float32(G)
    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G32)
    gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

    U = get_U_grid(n, Lbox, L, kernel)
    phi_K_raw = conv_fft(rho, U, zero_mode=True)

    # R3-7: Polarity reversal (try negative sign so kernel adds visibly)
    phi_K = (-1.0 * mu * G32 * phi_K_raw).astype(np.float32)
    gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    # DIAGNOSTIC: print a local force comparison at ~10 kpc (if resolvable)
    iz = n // 2
    try:
        # compute approximate index offset for ~10 kpc
        offset = int(round(10.0 / ( (2.0 * Lbox) / float(n) )))
        ix = n // 2 + offset
        iy = n // 2
        if 0 <= ix < n:
            b_fx = float(gx_b[ix, iy, iz])
            k_fx = float(gx_K[ix, iy, iz])
            print(f"[DIAGNOSTIC] L={L}, mu={mu} -> baryon_fx={b_fx:.4e}, kernel_fx={k_fx:.4e}")
    except Exception as e:
        print(f"[DIAGNOSTIC ERROR] {e}")

    # GAUGE: compose scalar squared magnitudes (vector addition allowed)
    g_total_sq = (gx_b[:, :, iz] + gx_K[:, :, iz])**2 + (gy_b[:, :, iz] + gy_K[:, :, iz])**2
    g_baryon_sq = gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2
    g_kernel_sq = gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2

    # radial projection
    R_centers, g_mean_total = radial_profile_2d(np.sqrt(g_total_sq), dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_baryons = radial_profile_2d(np.sqrt(g_baryon_sq), dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_kernel = radial_profile_2d(np.sqrt(g_kernel_sq), dx, R_obs_max, nbins=RADIAL_BINS)

    v_total = np.sqrt(np.maximum(R_centers * g_mean_total, 0.0))
    v_baryons = np.sqrt(np.maximum(R_centers * g_mean_baryons, 0.0))
    v_kernel = np.sqrt(np.maximum(R_centers * g_mean_kernel, 0.0))

    # cleanup heavy objects
    del rho, phi_b, phi_K_raw, phi_K, gx_b, gy_b, gx_K, gy_K, g_total_sq
    gc.collect()

    return R_centers, v_total, v_baryons, v_kernel

def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))

# ---- MAIN CONTROLLER ----

def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    download_and_extract_data()

    # Read only NGC3198 (fast mode)
    table_path = os.path.join("data", "galaxies.csv")
    gals = read_galaxy_table(table_path)

    print(f"Initializing Ananta Surveyor for {len(gals)} galaxy(ies) [SINGLE-GALAXY DEBUG MODE]")
    all_best_params = {}
    summary = []

    for i, gal in enumerate(gals):
        safe_name = sanitize_filename(gal["name"])
        print(f"\n[{i+1}/{len(gals)}] Surveying {safe_name}...")

        obs = try_read_observed_rc(gal["name"])
        if obs is None:
            print(f"  -> MISSING DATA for {gal['name']}")
            continue

        R_obs, V_obs = obs
        local_best = {"L": None, "mu": None, "mafe": 1e99, "kernel": None}

        for kernel in KERNELS:
            for L in L_LIST:
                for mu in MU_LIST:
                    result = predict_rc_for_params(gal, L, mu, kernel)
                    if result is None:
                        continue
                    R_pred, V_pred, _, _ = result
                    # interpolate predicted total onto observed radii
                    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
                    m = np.isfinite(Vp)
                    if np.any(m):
                        score = mafe(Vp[m], V_obs[m])
                        if score < local_best["mafe"]:
                            local_best = {"L": float(L), "mu": float(mu), "mafe": score, "kernel": kernel}
                    gc.collect()

        if local_best["L"] is None:
            print(f"  -> Fit Failed for {gal['name']}")
            continue

        all_best_params[gal["name"]] = local_best
        summary.append({"name": gal["name"], "mafe": local_best["mafe"], "L": local_best["L"], "mu": local_best["mu"]})
        print(f"  -> Best Fit: L={local_best['L']} kpc, mu={local_best['mu']} (Error: {local_best['mafe']:.4f})")

        # compute final decomposition and save figure
        final = predict_rc_for_params(gal, local_best["L"], local_best["mu"], local_best["kernel"])
        if final is None:
            continue
        R_pred, V_pred, V_b, V_k = final

        out = f"figs/rc_{safe_name}_best.png"
        plt.figure(figsize=(6, 4.5))
        plt.plot(R_pred, V_pred, "-", color="orange", lw=2, label="H1 Total")
        plt.plot(R_pred, V_b, ":", color="c", lw=1.5, label="Baryons")
        plt.plot(R_pred, V_k, "--", color="lime", lw=1.5, label=f"Kernel (L={local_best['L']}, mu={local_best['mu']})")
        plt.plot(R_obs, V_obs, "o", mec="black", mfc="white", ms=4, mew=0.6, label="Observed")
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — Best Fit")
        try:
            plt.xlim(0, float(np.nanmax(R_obs)) * 1.1)
        except Exception:
            pass
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        np.savetxt(
            f"results/rc_decomp_{safe_name}_best.csv",
            np.c_[R_pred, V_b, V_k, V_pred],
            delimiter=",",
            header="R_kpc,V_baryon,V_kernel,V_total",
            comments=""
        )

        gc.collect()

    # write summaries
    with open("results/all_galaxy_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2)
    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe,best_L,best_mu\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']},{row['L']},{row['mu']}\n")

if __name__ == "__main__":
    main()
