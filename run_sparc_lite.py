# run_sparc_lite.py — SPARC-lite loop (stars+gas) in physical km/s

import os, csv, json
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import two_component_disk
from src.kernels import U_plummer, U_exp_core
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# --------------------
# Knobs you can tweak
# --------------------
RADIAL_BINS = 30
KERNELS = ("plummer", "exp-core")

# Coarse search ranges (global fit across the sample)
L_LIST  = [2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0]      # kpc
MU_LIST = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]    # dimensionless


# --------------------
# Utilities
# --------------------
def choose_box_and_grid(R_obs_max, L):
    """
    Pick a simulation cube half-size (Lbox) and grid size (n) that cover the
    galaxy and the kernel tail. We aim for dx ~ 0.6–1.0 kpc.
    """
    target_half = max(1.5 * float(R_obs_max), 6.0 * float(L), 20.0)   # kpc
    Lbox = float(target_half)
    n = int(round(2 * Lbox / 0.8))   # ~0.8 kpc per cell
    n = max(64, min(n, 160))
    if n % 2 == 1:
        n += 1
    return Lbox, n


def build_U_grid(n, Lbox, L, kernel):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x*x + y*y + z*z)
    if kernel == "plummer":
        U = U_plummer(r, L)
    elif kernel == "exp-core":
        U = U_exp_core(r, L)
    else:
        raise ValueError("kernel must be 'plummer' or 'exp-core'")
    # zero the DC mode so only gradients matter
    U.flat[0] = 0.0
    return U


def radial_profile_2d(arr2d, dx, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5)**2 + (yy - cy + 0.5)**2) * dx
    rb = np.linspace(0, (n//2 - 1) * dx, nbins + 1)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins); prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        prof[i] = float(np.mean(arr2d[m])) if np.any(m) else np.nan
    return centers, prof


def read_galaxy_table(path_csv):
    out = []
    with open(path_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(x):
                try:
                    return float(x)
                except:
                    return 0.0
            rec = {
                "name":   row["name"],
                "Rd_star": num(row["Rd_star_kpc"]),
                "Mstar":   num(row["Mstar_Msun"]),
                "hz_star": num(row["hz_star_kpc"]),
                "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                "Mgas":    num(row.get("Mgas_Msun", "0")),
                "hz_gas":  num(row.get("hz_gas_kpc", "1.8")),
            }
            # If gas scale length missing/non-positive, set to 1.8 * Rd_star
            if rec["Rd_gas"] <= 0.0:
                rec["Rd_gas"] = 1.8 * rec["Rd_star"]
            out.append(rec)
    return out


def try_read_observed_rc(name):
    """
    Expected CSV at data/sparc/<NAME>_rc.csv with columns: R_kpc, V_kms
    """
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


# --------------------
# Core prediction
# --------------------
def predict_rc_for_params(gal, L, mu, kernel):
    """
    Given galaxy baryons + (L, mu, kernel), return (R_pred, V_pred) in km/s.
    The cube size and resolution are chosen from the observed extent.
    """
    # 1) choose box/grid from observed RC if present
    R_obs_max = 15.0
    obs = try_read_observed_rc(gal["name"])
    if obs is not None and np.size(obs[0]) > 0:
        R_obs_max = float(np.nanmax(obs[0]))
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    # 2) build baryon density on that grid
    rho, dx = two_component_disk(
        n, Lbox,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    # 3) Newtonian baryon potential and field
    phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=G)
    gx_b, gy_b, gz_b = gradient_from_phi(phi_b, Lbox)

    # 4) Kernel potential part: mu * G * conv(rho, U)
    U_grid = build_U_grid(n, Lbox, L, kernel)
    phi_K_raw = conv_fft(rho, U_grid, zero_mode=True)
    phi_K = mu * G * phi_K_raw
    gx_K, gy_K, gz_K = gradient_from_phi(phi_K, Lbox)

    # 5) total mid-plane circular speed
    iz = n // 2
    gmag = np.sqrt((gx_b[:, :, iz] + gx_K[:, :, iz])**2 +
                   (gy_b[:, :, iz] + gy_K[:, :, iz])**2)
    centers, gmean = radial_profile_2d(gmag, dx, nbins=RADIAL_BINS)
    vpred = np.sqrt(np.maximum(centers * gmean, 0.0))
    return centers, vpred


def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))


# --------------------
# Main experiment
# --------------------
def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    table_path = os.path.join("data", "galaxies.csv")
    if not os.path.exists(table_path):
        print("No data/galaxies.csv found. Add it and commit.")
        return
    gals = read_galaxy_table(table_path)
    if not gals:
        print("galaxies.csv is empty.")
        return

    # Global grid-search (L, mu, kernel) using galaxies that have observed RC
    best = {"L": None, "mu": None, "mafe": 1e99, "kernel": None}
    for kernel in KERNELS:
        for L in L_LIST:
            for mu in MU_LIST:
                scores = []
                for gal in gals:
                    obs = try_read_observed_rc(gal["name"])
                    if obs is None:
                        continue
                    R_obs, V_obs = obs
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

    # Per-galaxy plots + CSV exports with the best params
    summary = []
    for gal in gals:
        R_pred, V_pred = predict_rc_for_params(gal, best["L"], best["mu"], best["kernel"])
        obs = try_read_observed_rc(gal["name"])

        # Plot
        out_png = f"figs/rc_{gal['name']}_{best['kernel']}.png"
        plt.figure(figsize=(5, 4))
        plt.plot(R_pred, V_pred, "-", lw=2,
                 label=f'H1 pred (L={best["L"]:.1f} kpc, μ={best["mu"]:.2f})')
        if obs is not None:
            R_obs, V_obs = obs
            plt.plot(R_obs, V_obs, "o", ms=3, label="observed RC")
            Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
            m = np.isfinite(Vp)
            score = mafe(Vp[m], V_obs[m]) if np.any(m) else float("nan")
        else:
            score = float("nan")
            R_obs = np.array([])
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — {best['kernel']}")
        # Nice x-range: cover both curves
        xmax = float(np.nanmax([np.nanmax(R_pred), np.nanmax(R_obs) if R_obs.size else 0.0])) * 1.05
        plt.xlim(0.0, xmax if xmax > 0 else np.nanmax(R_pred))
        plt.legend()
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        summary.append({"name": gal["name"], "mafe": score})
        print("Saved", out_png)

        # CSVs
        np.savetxt(
            f"results/rc_pred_{gal['name']}_{best['kernel']}.csv",
            np.c_[R_pred, V_pred], delimiter=",",
            header="R_kpc,Vpred_kms", comments=""
        )
        if obs is not None:
            np.savetxt(
                f"results/rc_obs_{gal['name']}.csv",
                np.c_[R_obs, V_obs], delimiter=",",
                header="R_kpc,V_kms", comments=""
            )

    # Save summary files
    with open("results/sparc_lite_best.json", "w") as f:
        json.dump(best, f, indent=2)
    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']}\n")


if __name__ == "__main__":
    main()
