# run_sparc_lite.py — SPARC-lite loop (stars+gas), physical km/s

import os, csv, math
import numpy as np
import matplotlib.pyplot as plt

from src.baryons import two_component_disk
from src.kernels import U_plummer, U_exp_core
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- knobs (fast defaults for CI/local) ----
GRID_N  = 64
LBOX    = 20.0   # kpc (cube spans [-LBOX, +LBOX])
KERNELS = ("plummer", "exp-core")
L_LIST  = [4.0, 6.0, 8.0]   # kpc
MU_LIST = [0.8, 1.0, 1.2]   # dimensionless

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
    U.flat[0] = 0.0
    return U

def radial_profile_2d(arr2d, dx, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt((xx - cx + 0.5)**2 + (yy - cy + 0.5)**2) * dx
    rb = np.linspace(0, (n//2 - 1)*dx, nbins+1)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins); prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        prof[i] = float(np.mean(arr2d[m])) if np.any(m) else np.nan
    return centers, prof

def read_galaxy_table(path_csv):
    out = []
    with open(path_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # coerce numbers (blank -> 0.0)
            def num(x):
                try: return float(x)
                except: return 0.0
            out.append({
                "name": row["name"],
                "Rd_star": num(row["Rd_star_kpc"]),
                "Mstar":   num(row["Mstar_Msun"]),
                "hz_star": num(row["hz_star_kpc"]),
                "Rd_gas":  num(row.get("Rd_gas_kpc","0")),
                "Mgas":    num(row.get("Mgas_Msun","0")),
                "hz_gas":  num(row.get("hz_gas_kpc","0.3")),
            })
    return out

def try_read_observed_rc(name):
    # Expected path: data/sparc/<NAME>_rc.csv with columns: R_kpc,V_kms (and optional eV_kms)
    path = os.path.join("data","sparc",f"{name}_rc.csv")
    if not os.path.exists(path):
        return None
    R, V = [], []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            R.append(float(row["R_kpc"]))
            V.append(float(row["V_kms"]))
    return np.array(R), np.array(V)

def predict_rc_for_params(gal, L, mu, kernel):
    # Build baryon density
    rho, dx = two_component_disk(
        GRID_N, LBOX,
        Rd_star=gal["Rd_star"], Mstar=gal["Mstar"], hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],   Mgas=gal["Mgas"],   hz_gas=gal["hz_gas"]
    )
    # Newtonian baryon potential and field
    phi_b = phi_newtonian_from_rho(rho, LBOX, Gval=G)
    gx_b, gy_b, gz_b = gradient_from_phi(phi_b, LBOX)

    # Kernel potential part (μ G ∫ ρ U)
    U = build_U_grid(GRID_N, LBOX, L, kernel)
    phi_K_raw = conv_fft(rho, U, zero_mode=True)     # has units of Msun/kpc * (1?)
    phi_K = mu * G * phi_K_raw                        # km^2/s^2, same as φ_b
    gx_K, gy_K, gz_K = gradient_from_phi(phi_K, LBOX)

    # Total mid-plane ring-averaged RC
    iz = GRID_N // 2
    gmag = np.sqrt((gx_b[:,:,iz] + gx_K[:,:,iz])**2 +
                   (gy_b[:,:,iz] + gy_K[:,:,iz])**2)
    centers, gmean = radial_profile_2d(gmag, dx, nbins=30)
    vpred = np.sqrt(np.maximum(centers * gmean, 0.0))  # km/s
    return centers, vpred

def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))

def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1) read tiny galaxy list
    table_path = os.path.join("data","galaxies.csv")
    if not os.path.exists(table_path):
        print("No data/galaxies.csv found. Add it and commit.")
        return
    gals = read_galaxy_table(table_path)
    if not gals:
        print("galaxies.csv is empty.")
        return

    # 2) grid-search global (L, μ) minimizing median MAFE over galaxies with observed RC
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
                    # interpolate prediction at observed radii
                    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
                    m = np.isfinite(Vp)
                    if np.any(m):
                        scores.append(mafe(Vp[m], V_obs[m]))
                if scores:
                    med = float(np.median(scores))
                    if med < best["mafe"]:
                        best = {"L": float(L), "mu": float(mu), "mafe": med, "kernel": kernel}

    # If no observed RCs present, just pick defaults
    if best["L"] is None:
        best = {"L": 6.0, "mu": 1.0, "mafe": float("nan"), "kernel": "plummer"}
    print("Best (median MAFE):", best)

    # 3) make per-galaxy plots with best params (overlay obs if available)
    summary = []
    for gal in gals:
        R_pred, V_pred = predict_rc_for_params(gal, best["L"], best["mu"], best["kernel"])
        obs = try_read_observed_rc(gal["name"])
        out = f"figs/rc_{gal['name']}_{best['kernel']}.png"
        plt.figure(figsize=(5,4))
        plt.plot(R_pred, V_pred, '-', lw=2, label=f'H1 pred (L={best["L"]:.1f} kpc, μ={best["mu"]:.2f})')
        if obs is not None:
            R_obs, V_obs = obs
            plt.plot(R_obs, V_obs, 'o', ms=3, label='observed RC')
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
        # save the predicted curve for Excel: R and Vpred in km/s
        np.savetxt(
            f"results/rc_pred_{gal['name']}_{best['kernel']}.csv",
            np.c_[R_pred, V_pred],
            delimiter=",",
            header="R_kpc,Vpred_kms",
            comments=""
        )

        # if you provided observed RC, save it too
        if obs is not None:
            R_obs, V_obs = obs
            np.savetxt(
                f"results/rc_obs_{gal['name']}.csv",
                np.c_[R_obs, V_obs],
                delimiter=",",
                header="R_kpc,V_kms",
                comments=""
            )

       Commit: feat(results): export predicted/observed RC CSVs.
    # 4) save metrics
    import json
    with open("results/sparc_lite_best.json","w") as f:
        json.dump(best, f, indent=2)
    with open("results/sparc_lite_summary.csv","w") as f:
        f.write("name,mafe\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']}\n")

if __name__ == "__main__":
    main()
