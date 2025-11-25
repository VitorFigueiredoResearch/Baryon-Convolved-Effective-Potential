# gate_c_beta_sweep.py — R3-9 "The Final Diagnostic"
# Corrected for GitHub Actions file structure

import json, os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
# Use relative paths for GitHub environment
GATE_B_JSON = Path("results/gate_b_results.json")   
PARAMS_JSON = Path("results/all_galaxy_params.json")
OUT_DIR = Path("results/gate_c_figs")
OUT_JSON = Path("results/gate_c_results.json")
OUT_CSV  = Path("results/gate_c_summary.csv")

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Beta sweep parameters
BETA_MIN, BETA_MAX, BETA_STEP = 0.85, 1.15, 0.005
betas = np.arange(BETA_MIN, BETA_MAX + 1e-9, BETA_STEP)

# --- IMPORT ENGINE ---
try:
    from run_sparc_lite import predict_rc_for_params, try_read_observed_rc, read_galaxy_table
    print("Engine imported successfully.")
except Exception as e:
    print(f"ERROR importing engine: {e}")
    sys.exit(1)

# --- HELPERS ---
def mafe(v_obs, v_pred):
    v_obs, v_pred = np.asarray(v_obs), np.asarray(v_pred)
    mask = v_obs > 1e-8
    if mask.sum() == 0: return float('nan')
    return np.median(np.abs((v_pred[mask] - v_obs[mask]) / v_obs[mask]))

# --- LOAD DATA ---
if not GATE_B_JSON.exists():
    print(f"Skipping Gate C: {GATE_B_JSON} not found (Gate B didn't run?)")
    sys.exit(0)

with open(GATE_B_JSON) as f:
    gate_b = json.load(f)

# Get Gold Params (NGC3198)
if not PARAMS_JSON.exists():
    print("Warning: all_galaxy_params.json not found. Using defaults.")
    L_gold, mu_gold, kernel_gold = 50.0, 50.0, "ananta-hybrid"
else:
    with open(PARAMS_JSON) as f:
        p = json.load(f)
        # Try NGC3198, else first available
        target = p.get("NGC3198", list(p.values())[0])
        L_gold = float(target.get("L", 50.0))
        mu_gold = float(target.get("mu", 50.0))
        kernel_gold = target.get("kernel", "ananta-hybrid")

print(f"--- GATE C START ---")
print(f"Universal Key: L={L_gold}, mu={mu_gold}")

# --- MAIN LOOP ---
results = []
gal_table = read_galaxy_table("data/galaxies.csv")

for item in gate_b:
    name = item.get("galaxy")
    if not name: continue
    
    print(f"Sweeping {name}...")
    
    # 1. Get Observed Data
    obs = try_read_observed_rc(name)
    if obs is None: continue
    R_obs, V_obs = obs

    # 2. Get Prediction (Force calculation using Gold Params)
    gal_meta = next((g for g in gal_table if g["name"] == name), None)
    if not gal_meta: continue
    
    pred = predict_rc_for_params(gal_meta, L_gold, mu_gold, kernel_gold)
    if pred is None: continue
    R_pred, V_pred, _, _ = pred
    
    # 3. Interpolate
    V_pred_interp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
    valid_mask = np.isfinite(V_pred_interp) & np.isfinite(V_obs)
    
    if not np.any(valid_mask): continue
    
    # 4. Sweep Beta (Scale Factor)
    baseline_mafe = mafe(V_obs[valid_mask], V_pred_interp[valid_mask])
    
    best_beta, best_mafe = 1.0, baseline_mafe
    
    for b in betas:
        score = mafe(V_obs[valid_mask], V_pred_interp[valid_mask] * b)
        if score < best_mafe:
            best_mafe = score
            best_beta = float(b)
            
    improvement = (baseline_mafe - best_mafe) / baseline_mafe if baseline_mafe > 0 else 0.0
    
    print(f"  -> Best Beta: {best_beta:.3f} (Imp: {improvement:.1%})")
    
    results.append({
        "galaxy": name,
        "baseline_mafe": baseline_mafe,
        "best_beta": best_beta,
        "best_mafe": best_mafe,
        "improvement": improvement
    })

    # 5. Plot
    plt.figure(figsize=(6,4))
    plt.plot(R_obs, V_obs, 'ko', label='Observed', ms=4)
    plt.plot(R_obs, V_pred_interp, 'r--', label='Universal (Beta=1.0)')
    plt.plot(R_obs, V_pred_interp * best_beta, 'g-', label=f'Scaled (Beta={best_beta:.2f})')
    plt.title(f"{name} - Gate C (Beta Sweep)")
    plt.xlabel("R [kpc]"); plt.ylabel("V [km/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_gateC.png", dpi=100)
    plt.close()

# --- SAVE SUMMARY ---
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("Gate C Complete.")
