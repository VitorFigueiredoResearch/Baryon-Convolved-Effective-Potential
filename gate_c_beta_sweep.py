#!/usr/bin/env python3
"""
Gate C: Per-galaxy beta sweep.

Usage:
  - Place this file at repo root.
  - Run: python gate_c_beta_sweep.py
Outputs (local):
  - /mnt/data/gate_c_results.json
  - /mnt/data/gate_c_summary.csv
  - /mnt/data/gate_c_figs/<GALNAME>_gateC.png

Notes:
  - Script first tries to import project functions:
      from run_sparc_lite import predict_rc_for_params, try_read_observed_rc, read_galaxy_table
    If those are missing, it will try fallback data sources (CSV files).
  - Gate B results input: /mnt/data/gate_b_results.json
"""

import json, os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths (adjust if your layout is different)
GATE_B_JSON = Path("/mnt/data/gate_b_results.json")   # your Gate B output (exists)
PARAMS_JSON = Path("/mnt/data/all_galaxy_params.json")  # universal params file (if present)
OUT_DIR = Path("/mnt/data/gate_c_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path("/mnt/data/gate_c_results.json")
OUT_CSV  = Path("/mnt/data/gate_c_summary.csv")

# Beta sweep parameters
BETA_MIN = 0.85
BETA_MAX = 1.15
BETA_STEP = 0.005
betas = np.arange(BETA_MIN, BETA_MAX + 1e-9, BETA_STEP)

# Helper: median absolute fractional error
def mafe(v_obs, v_pred):
    v_obs = np.asarray(v_obs)
    v_pred = np.asarray(v_pred)
    # avoid divide-by-zero: require v_obs>0; if zeros, ignore those points
    mask = v_obs > 1e-8
    if mask.sum() == 0:
        return float('nan')
    return np.median(np.abs((v_pred[mask] - v_obs[mask]) / v_obs[mask]))

# Try to import project functions (best case)
predict_fn = None
read_obs_fn = None
read_table_fn = None
try:
    # These imports depend on your repo layout; adapt names if different
    from run_sparc_lite import predict_rc_for_params, try_read_observed_rc, read_galaxy_table
    predict_fn = predict_rc_for_params
    read_obs_fn = try_read_observed_rc
    read_table_fn = read_galaxy_table
    print("Imported predict_rc_for_params and try_read_observed_rc from run_sparc_lite.")
except Exception as e:
    print("Project import fallback active (run_sparc_lite functions not found):", e)

# Load gate B list
if not GATE_B_JSON.exists():
    print("ERROR: Gate B results not found at", GATE_B_JSON)
    sys.exit(1)

gate_b = json.load(open(GATE_B_JSON))

# Determine universal params (NGC3198) from params JSON if present
univ_params = {}
if PARAMS_JSON.exists():
    try:
        p = json.load(open(PARAMS_JSON))
        if "NGC3198" in p:
            univ_params = p["NGC3198"]
        else:
            # try top-level keys
            univ_params = next(iter(p.values()))
    except Exception as e:
        print("Warning: could not parse all_galaxy_params.json:", e)

# fallback: try reading L and mu from gate_b entries (if saved)
if not univ_params:
    # gate_b may contain ngc params or global params
    # search for NGC3198 entry
    for item in gate_b:
        if item.get("galaxy","").lower().startswith("ngc3198"):
            univ_params = item.get("used_params", {}) or {}
            break

# final fallback defaults
L_default = float(univ_params.get("L", 50.0))
mu_default = float(univ_params.get("mu", 50.0))
print(f"Using universal params L={L_default}, mu={mu_default}")

results = []

# Helper to read observed RC by several methods
def get_observed_rc(galname):
    # Try repo function first
    if read_obs_fn is not None:
        try:
            R_obs, V_obs = read_obs_fn(galname)  # expecting two arrays
            return np.array(R_obs), np.array(V_obs)
        except Exception:
            pass
    # fallback: look for common CSV names in project/data or /mnt/data
    candidates = [
        Path(f"data/observed/{galname}.csv"),
        Path(f"data/observed/{galname}.txt"),
        Path(f"data/{galname}.csv"),
        Path(f"/mnt/data/{galname}.csv"),
        Path(f"/mnt/data/observed/{galname}.csv")
    ]
    for c in candidates:
        if c.exists():
            try:
                import pandas as pd
                df = pd.read_csv(c)
                # heuristics for columns
                cols = df.columns.str.lower()
                # common names: R, V or radius, velocity
                if 'r' in cols or 'radius' in cols:
                    rcol = df.columns[[('r' in s or 'radius' in s) for s in cols]][0]
                else:
                    rcol = df.columns[0]
                if 'v' in cols or 'velocity' in cols or 'vel' in cols:
                    vcol = df.columns[[('v' in s or 'velocity' in s or 'vel' in s) for s in cols]][0]
                else:
                    vcol = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                return df[rcol].to_numpy(), df[vcol].to_numpy()
            except Exception:
                pass
    return None, None

# Helper to get predicted RC: try to reuse saved predictions in gate_b JSON, else call predict_fn
def get_predicted_rc_from_gateb(item):
    # gate_b may have fields like 'pred_R' and 'pred_V' or 'prediction' stored
    preds = item.get("prediction") or item.get("predicted") or {}
    if preds:
        R = np.array(preds.get("R", preds.get("r", [])))
        V = np.array(preds.get("V", preds.get("v", [])))
        if len(R) and len(V):
            return R, V
    # maybe gate_b item saved arrays directly
    if "R_pred" in item and "V_pred" in item:
        return np.array(item["R_pred"]), np.array(item["V_pred"])
    return None, None

# If predict_fn is available, wrapper to compute predicted RC from universal params
def compute_pred_rc(gal_meta, L=L_default, mu=mu_default):
    if predict_fn is None:
        return None, None
    try:
        # expected signature: predict_rc_for_params(gal_meta_or_name, L, mu, kernel_name)
        out = predict_fn(gal_meta, L, mu, kernel=None)
        # accept either tuple (R, V) or dict
        if isinstance(out, tuple) and len(out) >= 2:
            return np.array(out[0]), np.array(out[1])
        if isinstance(out, dict):
            return np.array(out.get("R")), np.array(out.get("V"))
    except Exception as e:
        print("predict_fn call failed for", gal_meta, e)
    return None, None

# Iterate galaxies listed in gate_b JSON
for item in gate_b:
    galname = item.get("galaxy") or item.get("name")
    if not galname:
        continue
    print("Processing", galname)
    # observed
    R_obs, V_obs = get_observed_rc(galname)
    if R_obs is None or V_obs is None:
        print("  [SKIP] observed RC not found for", galname)
        results.append({
            "galaxy": galname,
            "status": "skip_no_observed",
        })
        continue

    # predicted: try gate_b stored, then repo predict function
    R_pred, V_pred = get_predicted_rc_from_gateb(item)
    if R_pred is None or V_pred is None:
        # attempt to call predict function if available
        R_pred, V_pred = compute_pred_rc(galname, L_default, mu_default)
    if R_pred is None or V_pred is None:
        print("  [SKIP] predicted RC not found for", galname)
        results.append({
            "galaxy": galname,
            "status": "skip_no_prediction",
        })
        continue

    # Interpolate predicted to observed R points if necessary
    try:
        # if R_pred is not same as R_obs, interpolate
        if not np.allclose(R_pred, R_obs):
            V_pred_interp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
        else:
            V_pred_interp = V_pred
    except Exception as e:
        print("  interpolation failed:", e)
        results.append({"galaxy": galname, "status": "error_interp"})
        continue

    # compute baseline MAFE
    baseline_mafe = mafe(V_obs, V_pred_interp)

    # sweep betas
    best_beta = None
    best_mafe = float('inf')
    best_vpred = None
    for b in betas:
        v_try = V_pred_interp * b
        m = mafe(V_obs, v_try)
        if np.isnan(m):
            continue
        if m < best_mafe:
            best_mafe = m
            best_beta = float(b)
            best_vpred = v_try.copy()

    improvement = None
    if (not math.isnan(baseline_mafe)) and best_mafe is not None:
        try:
            improvement = float((baseline_mafe - best_mafe) / baseline_mafe)
        except Exception:
            improvement = None

    # save plot
    try:
        plt.figure(figsize=(6,4))
        plt.plot(R_obs, V_obs, 'o', label='observed', markersize=5)
        plt.plot(R_pred, V_pred, '-', alpha=0.6, label='pred (orig)', linewidth=2)
        plt.plot(R_obs, best_vpred, '--', label=f'pred (beta={best_beta:.3f})', linewidth=2)
        plt.xlabel('R [kpc]'); plt.ylabel('V [km/s]')
        plt.title(f'{galname} - Gate C')
        plt.legend()
        plt.tight_layout()
        fname = OUT_DIR / f"{galname.replace('/','_')}_gateC.png"
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print("  warning: could not save plot for", galname, e)

    rec = {
        "galaxy": galname,
        "status": "ok",
        "N_points": int(np.sum(~np.isnan(V_obs))),
        "baseline_mafe": float(baseline_mafe),
        "best_beta": float(best_beta) if best_beta is not None else None,
        "best_mafe": float(best_mafe) if best_mafe is not None else None,
        "improvement_fraction": float(improvement) if improvement is not None else None
    }
    results.append(rec)
    print(f"  baseline_mafe={baseline_mafe:.4f}, best_beta={best_beta}, best_mafe={best_mafe:.4f}, improvement={improvement:.3f}")

# write results
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
# also write a CSV summary
import csv
with open(OUT_CSV, "w", newline='') as csvf:
    w = csv.writer(csvf)
    w.writerow(["galaxy","status","N_points","baseline_mafe","best_beta","best_mafe","improvement_fraction"])
    for r in results:
        w.writerow([r.get("galaxy"), r.get("status"), r.get("N_points"), r.get("baseline_mafe"), r.get("best_beta"), r.get("best_mafe"), r.get("improvement_fraction")])

print("Gate C finished. Results:", OUT_JSON, OUT_CSV)
