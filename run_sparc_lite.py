# run_sparc_lite.py — Cleaned R3-7 "Hardened Surveyor"
# Ready-to-paste, minimal changes from your last version.
# Notes:
#  - Set TARGET_GALAXY = "NGC3198" to debug one galaxy
#  - Set TARGET_GALAXY = None to run the whole fleet (or CSV)

import os
import csv
import json
import gc
import urllib.request
import zipfile
import re
import numpy as np
import matplotlib.pyplot as plt

# Local physics modules (must be importable from repo root)
from src.kernels import U_plummer, U_exp_core, U_ananta_hybrid
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- CONFIG ----
RADIAL_BINS = 30

# Kernel search lists 
KERNELS = ("ananta-hybrid",)
L_LIST = [10.0, 30.0, 50.0, 80.0, 120.0, 200.0]
MU_LIST = [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]

# Target: set to a string to run ONE galaxy (fast debug), or None for all from CSV
TARGET_GALAXY = None   # e.g. "NGC3198" for fast debug

# If you need to flip kernel sign for testing (polarity fix), change to -1.0
POLARITY_SIGN = 1.0

# ---- Backup fleet (fallback if CSV not present or empty) ----
NIGHTMARE_FLEET = [
    # --- The Original Nightmare Fleet ---
{"name": "NGC0628", "Rd_star": 2.92, "Mstar": 3.02e10, "hz_star": 0.61, "Rd_gas": 4.38, "Mgas": 1.13e10, "hz_gas": 0.15},
{"name": "NGC0925", "Rd_star": 3.30, "Mstar": 1.41e10, "hz_star": 0.43, "Rd_gas": 4.95, "Mgas": 2.53e10, "hz_gas": 0.15},
{"name": "NGC3319", "Rd_star": 2.18, "Mstar": 6.46e9, "hz_star": 0.33, "Rd_gas": 3.27, "Mgas": 4.09e9, "hz_gas": 0.15},
{"name": "NGC4321", "Rd_star": 2.78, "Mstar": 4.36e10, "hz_star": 0.64, "Rd_gas": 4.17, "Mgas": 1.68e10, "hz_gas": 0.15},
{"name": "NGC4736", "Rd_star": 0.93, "Mstar": 2.94e10, "hz_star": 0.15, "Rd_gas": 1.40, "Mgas": 5.67e9, "hz_gas": 0.15},
{"name": "NGC4826", "Rd_star": 0.89, "Mstar": 3.33e10, "hz_star": 0.13, "Rd_gas": 1.34, "Mgas": 5.11e9, "hz_gas": 0.15},
{"name": "NGC5035", "Rd_star": 3.97, "Mstar": 3.11e10, "hz_star": 0.63, "Rd_gas": 5.96, "Mgas": 1.94e10, "hz_gas": 0.15},
{"name": "NGC5713", "Rd_star": 1.83, "Mstar": 2.71e10, "hz_star": 0.30, "Rd_gas": 2.75, "Mgas": 1.33e10, "hz_gas": 0.15},
{"name": "NGC6949", "Rd_star": 2.41, "Mstar": 1.77e10, "hz_star": 0.32, "Rd_gas": 3.62, "Mgas": 1.08e10, "hz_gas": 0.15},
{"name": "NGC7790", "Rd_star": 3.77, "Mstar": 3.63e10, "hz_star": 0.49, "Rd_gas": 5.66, "Mgas": 2.14e10, "hz_gas": 0.15},
{"name": "NGC0108", "Rd_star": 3.00, "Mstar": 3.06e10, "hz_star": 0.45, "Rd_gas": 4.50, "Mgas": 1.72e10, "hz_gas": 0.15},
{"name": "NGC0550", "Rd_star": 2.36, "Mstar": 1.34e10, "hz_star": 0.35, "Rd_gas": 3.54, "Mgas": 6.95e9, "hz_gas": 0.15},
{"name": "NGC1325", "Rd_star": 3.41, "Mstar": 2.35e10, "hz_star": 0.52, "Rd_gas": 5.12, "Mgas": 1.57e10, "hz_gas": 0.15},
{"name": "NGC1808", "Rd_star": 1.44, "Mstar": 2.57e10, "hz_star": 0.18, "Rd_gas": 2.16, "Mgas": 8.12e9, "hz_gas": 0.15},
{"name": "NGC2599", "Rd_star": 3.64, "Mstar": 2.73e10, "hz_star": 0.56, "Rd_gas": 5.46, "Mgas": 1.84e10, "hz_gas": 0.15},
{"name": "NGC3351", "Rd_star": 1.73, "Mstar": 2.90e10, "hz_star": 0.28, "Rd_gas": 2.60, "Mgas": 6.96e9, "hz_gas": 0.15},
{"name": "NGC3621", "Rd_star": 4.76, "Mstar": 2.12e10, "hz_star": 0.80, "Rd_gas": 7.14, "Mgas": 2.60e10, "hz_gas": 0.15},
{"name": "NGC4737", "Rd_star": 2.60, "Mstar": 2.40e10, "hz_star": 0.39, "Rd_gas": 3.90, "Mgas": 1.42e10, "hz_gas": 0.15},
{"name": "NGC7217", "Rd_star": 1.58, "Mstar": 3.67e10, "hz_star": 0.22, "Rd_gas": 2.37, "Mgas": 5.87e9, "hz_gas": 0.15},
{"name": "NGC7817", "Rd_star": 3.01, "Mstar": 2.84e10, "hz_star": 0.44, "Rd_gas": 4.52, "Mgas": 1.58e10, "hz_gas": 0.15},
{"name": "Holmberg II", "Rd_star": 0.88, "Mstar": 5.91e8, "hz_star": 0.05, "Rd_gas": 1.32, "Mgas": 3.68e9, "hz_gas": 0.15},
{"name": "Holmberg I", "Rd_star": 0.63, "Mstar": 2.76e8, "hz_star": 0.03, "Rd_gas": 0.94, "Mgas": 1.92e9, "hz_gas": 0.15},
{"name": "DDO047", "Rd_star": 0.99, "Mstar": 4.71e8, "hz_star": 0.07, "Rd_gas": 1.49, "Mgas": 2.49e9, "hz_gas": 0.15},
{"name": "DDO050", "Rd_star": 0.96, "Mstar": 5.34e8, "hz_star": 0.06, "Rd_gas": 1.44, "Mgas": 2.88e9, "hz_gas": 0.15},
{"name": "DDO133", "Rd_star": 0.90, "Mstar": 2.84e8, "hz_star": 0.05, "Rd_gas": 1.35, "Mgas": 1.91e9, "hz_gas": 0.15},
{"name": "DDO190", "Rd_star": 0.85, "Mstar": 2.33e8, "hz_star": 0.04, "Rd_gas": 1.28, "Mgas": 1.74e9, "hz_gas": 0.15},
{"name": "KKR25", "Rd_star": 0.61, "Mstar": 1.75e8, "hz_star": 0.03, "Rd_gas": 0.92, "Mgas": 1.00e9, "hz_gas": 0.15},
{"name": "NGC3106", "Rd_star": 3.67, "Mstar": 4.52e10, "hz_star": 0.55, "Rd_gas": 5.51, "Mgas": 1.94e10, "hz_gas": 0.15},
{"name": "UGC01281", "Rd_star": 8.0, "Mstar": 8.00e9, "hz_star": 1.92, "Rd_gas": 12.0, "Mgas": 1.06e9, "hz_gas": 0.15},
{"name": "UGC04278", "Rd_star": 7.0, "Mstar": 7.00e9, "hz_star": 0.70, "Rd_gas": 10.5, "Mgas": 9.31e8, "hz_gas": 0.15},
{"name": "UGC06917", "Rd_star": 9.0, "Mstar": 1.88e10, "hz_star": 1.80, "Rd_gas": 13.5, "Mgas": 1.20e9, "hz_gas": 0.15},
{"name": "UGC07089", "Rd_star": 8.0, "Mstar": 3.35e9, "hz_star": 0.71, "Rd_gas": 12.0, "Mgas": 1.06e9, "hz_gas": 0.15},
{"name": "UGC07524", "Rd_star": 9.0, "Mstar": 9.00e9, "hz_star": 0.50, "Rd_gas": 13.5, "Mgas": 1.20e9, "hz_gas": 0.15},
]

# ---- UTILITIES ----
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name)

# ---- DATA DOWNLOAD ----
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
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"!!! DOWNLOAD ERROR: {e}")

# ---- PHYSICS GRID & PROFILES ----
def safe_two_component_disk(n, Lbox, Rd_star, Mstar, hz_star, Rd_gas, Mgas, hz_gas):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x * x + y * y)
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
        rho0 = M / (4 * np.pi * Rd**2 * hz)
        return (rho0 * radial * vertical).astype(np.float32)

    rho_star = get_rho(Mstar, Rd_star, hz_star)
    rho_gas = get_rho(Mgas, Rd_gas, hz_gas)
    return rho_star + rho_gas, dx

def choose_box_and_grid(R_obs_max, L):
    target_half = max(1.5 * R_obs_max, 4.0 * L, 20.0)
    Lbox = float(target_half)
    n = int(np.clip(round(2 * Lbox / 0.5), 64, 320))
    if n % 2 == 1:
        n += 1
    return Lbox, n

def build_U_grid(n, Lbox, L, kernel, beta=1.0):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    if kernel == "plummer":
        U = U_plummer(r, L)
        # finite centre for density kernels
        U.flat[0] = 1.0 / max(1e-6, float(L))
    elif kernel == "exp-core":
        U = U_exp_core(r, L)
        U.flat[0] = 1.0 / max(1e-6, (float(L) * 2.718))
    elif kernel == "ananta-hybrid":
        U = U_ananta_hybrid(r, L, beta=beta)
    else:
        raise ValueError("kernel error")

    return U.astype(np.float32)

U_CACHE = {}
def get_U_grid(n, Lbox, L, kernel, beta=1.0):
    key = (kernel, float(L), int(n), round(float(Lbox), 2), float(beta))
    if key not in U_CACHE:
        U_CACHE[key] = build_U_grid(n, Lbox, L, kernel, beta=beta)
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
    rb = np.linspace(0, max_r * 1.1, nbins + 1).astype(np.float32)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins, dtype=np.float32)
    prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        if np.any(m):
            prof[i] = np.mean(arr2d[m])
    return centers, fill_nans(prof)

# ---- IO ----
def read_galaxy_table(path_csv):
    out = []
    if os.path.exists(path_csv):
        try:
            with open(path_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # --- FAST MODE FILTER: only keep the TARGET if it is set ---
                    if TARGET_GALAXY and row["name"].strip() != TARGET_GALAXY:
                        continue

                    def num(x):
                        try: return float(x)
                        except: return 0.0

                    g = {
                        "name":    row["name"].strip(),
                        "Rd_star": num(row.get("Rd_star_kpc", 0)),
                        "Mstar":   num(row.get("Mstar_Msun", 0)),
                        "hz_star": num(row.get("hz_star_kpc", "0.3")),
                        "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                        "Mgas":    num(row.get("Mgas_Msun", "0")),
                        "hz_gas":  num(row.get("hz_gas_kpc", "0.15")),
                    }
                    if g["Rd_gas"] <= 0:
                        g["Rd_gas"] = 1.8 * (g["Rd_star"] if g["Rd_star"] > 0 else 1.0)
                    out.append(g)
        except Exception as e:
            print(f"Note: Error reading CSV ({e}).")

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

    R, V = [], []
    try:
        with open(file_to_read, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Rad"):
                    continue
                if is_dat:
                    parts = line.split()
                else:
                    parts = line.split(',')
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
def predict_rc_for_params(gal, L, mu, kernel, beta=1.0):
    """
    Compute predicted rotation curve for one galaxy.
    Added: beta (global amplitude correction) forwarded to kernel grid.
    """

    obs = try_read_observed_rc(gal["name"])
    if obs is None or obs[0].size == 0:
        return None

    R_obs_max = float(np.nanmax(obs[0]))
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    rho, dx = safe_two_component_disk(
        n,
        Lbox,
        Rd_star=gal["Rd_star"],
        Mstar=gal["Mstar"],
        hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],
        Mgas=gal["Mgas"],
        hz_gas=gal["hz_gas"]
    )

    G32 = np.float32(G)

    # --- Baryonic Newtonian potential ---
    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G32)
    gx_b, gy_b, _ = gradient_from_phi(phi_b, Lbox)

    # --- Kernel: build and compute safely (defensive) ---
    # PATCH: pass beta into the kernel grid builder
    U = get_U_grid(n, Lbox, L, kernel, beta=beta)

    phi_K_raw = None
    phi_K = None
    gx_K = np.zeros_like(gx_b, dtype=np.float32)
    gy_K = np.zeros_like(gy_b, dtype=np.float32)

    try:
        # convolution -> raw kernel potential
        phi_K_raw = conv_fft(rho, U, zero_mode=True)

        # Apply polarity sign and coupling
        phi_K = (POLARITY_SIGN * mu * G32 * phi_K_raw).astype(np.float32)

        # gradients -> forces from kernel
        gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    except Exception as e:
        print(f"[ERROR] Kernel computation failed for L={L}, mu={mu}: {e}")
        # leave gx_K, gy_K as zeros

    # --- Polarity diagnostic ---
    iz = n // 2
    try:
        offset = int(round(10.0 / max(1e-6, (2.0 * Lbox) / float(n))))
        ix = n // 2 + offset
        iy = n // 2

        if 0 <= ix < n:
            _b = float(gx_b[ix, iy, iz])
            _k = float(gx_K[ix, iy, iz])
            if (_b > 0 and _k < 0) or (_b < 0 and _k > 0):
                print(f"[WARNING] Polarity mismatch at L={L}, mu={mu}: "
                      f"baryon={_b:.3e}, kernel={_k:.3e}")

    except Exception:
        pass

    # RETURN expected values (handled elsewhere in full pipeline)
    # Not included here because you only gave the middle block.
    # Your full file should already have the R_pred, V_pred logic below.


    # Combine vectors (vector sum) then convert to radial profile magnitudes
    g_total_sq = (gx_b[:, :, iz] + gx_K[:, :, iz])**2 + (gy_b[:, :, iz] + gy_K[:, :, iz])**2
    g_baryon_sq = gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2
    g_kernel_sq = gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2

    R_centers, g_mean_total = radial_profile_2d(np.sqrt(g_total_sq), dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_baryons = radial_profile_2d(np.sqrt(g_baryon_sq), dx, R_obs_max, nbins=RADIAL_BINS)
    _, g_mean_kernel = radial_profile_2d(np.sqrt(g_kernel_sq), dx, R_obs_max, nbins=RADIAL_BINS)

    v_total = np.sqrt(np.maximum(R_centers * g_mean_total, 0.0))
    v_baryons = np.sqrt(np.maximum(R_centers * g_mean_baryons, 0.0))
    v_kernel = np.sqrt(np.maximum(R_centers * g_mean_kernel, 0.0))

    # Safe cleanup (avoids UnboundLocalError)
    for v in ("rho", "phi_b", "phi_K_raw", "phi_K",
              "gx_b", "gy_b", "gx_K", "gy_K", "g_total_sq"):
        if v in locals():
            try:
                del locals()[v]
            except Exception:
                pass

    gc.collect()
    return R_centers, v_total, v_baryons, v_kernel

def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))

# ---- MAIN ----
def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    download_and_extract_data()

    table_path = os.path.join("data", "galaxies.csv")
    gals = read_galaxy_table(table_path)

    print(f"Initializing Ananta Surveyor for {len(gals)} galaxies...")
    mode = f"SINGLE({TARGET_GALAXY})" if TARGET_GALAXY is not None else "FULL"
    print(f"Mode: {mode}")

    all_best_params = {}
    summary = []

    for i, gal in enumerate(gals):
        safe_name = sanitize_filename(gal['name'])
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
                    res = predict_rc_for_params(gal, L, mu, kernel, beta=1.15)
                    if res is None:
                        continue
                    R_pred, V_pred, _, _ = res
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

        final_res = predict_rc_for_params(
            gal,
            local_best["L"],
            local_best["mu"],
            local_best["kernel"],
            beta=1.15
        )

        if final_res is None:
            continue
        R_pred, V_pred, V_b, V_k = final_res

        out = os.path.join("figs", f"rc_{safe_name}_best.png")
        plt.figure(figsize=(5, 4))
        plt.plot(R_pred, V_pred, "-", color='orange', lw=2, label='H1 Total')
        plt.plot(R_pred, V_b, ':', color='cyan', lw=1.5, label='Baryons')
        plt.plot(R_pred, V_k, '--', color='lime', lw=1.5, label=f'Kernel (L={local_best["L"]}, mu={local_best["mu"]})')
        if obs is not None:
            plt.plot(R_obs, V_obs, "o", color='white', mec='black', ms=4, mew=0.5, label="Observed")
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — Best Fit")
        if obs is not None:
            plt.xlim(0, np.nanmax(R_obs) * 1.1)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        np.savetxt(
            os.path.join("results", f"rc_decomp_{safe_name}_best.csv"),
            np.c_[R_pred, V_b, V_k, V_pred],
            delimiter=",", header="R_kpc,V_baryon,V_kernel,V_total", comments=""
        )

        del R_pred, V_pred, V_b, V_k
        gc.collect()

    with open("results/all_galaxy_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2)
    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe,best_L,best_mu\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']},{row['L']},{row['mu']}\n")

if __name__ == "__main__":
    main()
