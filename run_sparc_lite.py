# run_sparc_lite.py — SPARC-lite loop (stars+gas) in physical km/s

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import two_component_disk
from src.kernels import U_plummer, U_exp_core
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- knobs (fast defaults) ----
GRID_N = 96
LBOX   = 80.0   # kpc (cube spans [-LBOX, +LBOX])
RADIAL_BINS = 30
KERNELS = ("plummer", "exp-core")

# coarse grid to search
L_LIST  = [2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0]
MU_LIST = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]


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
    # zero DC mode (absolute level of potential is irrelevant)
    U.flat[0] = 0.0
    return U


def radial_profile_2d(arr2d, dx, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) * dx
    rb = np.linspace(0, (n // 2 - 1) * dx, nbins + 1)
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
            d = {
                "name":    row["name"],
                "Rd_star": num(row["Rd_star_kpc"]),
                "Mstar":   num(row["Mstar_Msun"]),
                "hz_star": num(row["hz_star_kpc"]),
                "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                "Mgas":    num(row.get("Mgas_Msun", "0")),
                "hz_gas":  num(row.get("hz_gas_kpc", "0.3")),
            }
            # if gas disk scalelength is not provided, tie it to stars
            if d["Rd_gas"] <= 0:
                d["Rd_gas"] = 1.8 * d["Rd_star"]
            out.append(d)
    return out


def try_read_observed_rc(name):
    """Reads data/sparc/<NAME>_rc.csv with columns: R_kpc,V_kms."""
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


def predict_rc_for_params(gal, L, mu, U_grid):
    """Return (R_pred, V_pred) for one galaxy at given (L, mu) and kernel grid."""
    # baryon density and grid spacing
    rho, dx = two_component_disk(
        GRID_N, LBOX,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )

    # Newtonian field from baryons
    phi_b = phi_newtonian_from_rho(rho, LBOX, Gval=G)
    gx_b, gy_b, gz_b = gradient_from_phi(phi_b, LBOX)

    # Kernel potential part: phi_K = mu * G * conv(rho, U)
    phi_K_raw = conv_fft(rho, U_grid, zero_mode=True)
    phi_K = mu * G * phi_K_raw
    gx_K, gy_K, gz_K = gradient_from_phi(phi_K, LBOX)

    # mid-plane total acceleration magnitude
    iz = GRID_N // 2
    gmag = np.sqrt((gx_b[:, :, iz] + gx_K[:, :, iz]) ** 2 +
                   (gy_b[:, :, iz] + gy_K[:, :, iz]) ** 2)

    # ring-average and convert to circular speed
    R_centers, gmean = radial_profile_2d(gmag, dx, nbins=RADIAL_BINS)
    V_pred = np.sqrt(np.maximum(R_centers * gmean, 0.0))  # km/s
    return R_centers, V_pred


def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))


def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1) read galaxy list
    table_path = os.path.join("data", "galaxies.csv")
    if not os.path.exists(table_path):
        print("No data/galaxies.csv found. Add it and commit.")
        return
    gals = read_galaxy_table(table_path)
    if not gals:
        print("galaxies.csv is empty.")
        return

    # 2) precompute potential kernels
    print("Pre-calculating potential kernels (U_grid)...")
    U_grid_cache = {}
    for kernel in KERNELS:
        for L in L_LIST:
            U_grid_cache[(kernel, L)] = build_U_grid(GRID_N, LBOX, L, kernel)
    print("...done.")

    # 3) grid-search (L, mu, kernel) minimizing median MAFE over galaxies with obs RC
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
                    U_grid = U_grid_cache[(kernel, L)]
                    R_pred, V_pred = predict_rc_for_params(gal, L, mu, U_grid)
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

    # 4) per-galaxy plots + CSV exports with best params
    summary = []
    for gal in gals:
        U_grid = U_grid_cache[(best["kernel"], best["L"])]
        R_pred, V_pred = predict_rc_for_params(gal, best["L"], best["mu"], U_grid)
        obs = try_read_observed_rc(gal["name"])

        # plot
        out = f"figs/rc_{gal['name']}_{best['kernel']}.png"
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
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — {best['kernel']}")
        plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        summary.append({"name": gal["name"], "mafe": score})
        print("Saved", out)

        # export predicted curve
        np.savetxt(
            f"results/rc_pred_{gal['name']}_{best['kernel']}.csv",
            np.c_[R_pred, V_pred],
            delimiter=",",
            header="R_kpc,Vpred_kms",
            comments=""
        )
        # export observed curve if present
        if obs is not None:
            np.savetxt(
                f"results/rc_obs_{gal['name']}.csv",
                np.c_[R_obs, V_obs],
                delimiter=",",
                header="R_kpc,V_kms",
                comments=""
            )

    # 5) save metrics
    with open("results/sparc_lite_best.json", "w") as f:
        json.dump(best, f, indent=2)

    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']}\n")


if __name__ == "__main__":
    main()

