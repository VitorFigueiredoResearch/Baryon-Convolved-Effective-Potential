# gate_b_zero_refit.py
# Gate B: Zero-Refit Test
# Usage: python gate_b_zero_refit.py
# Expects: repo root with run_sparc_lite.py
# Writes: results/gate_b_results.json and figs/*_gateB.png

import json, numpy as np, os, sys
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import helper functions
try:
    from run_sparc_lite import predict_rc_for_params, read_galaxy_table, try_read_observed_rc
except Exception as e:
    print("ERROR importing run_sparc_lite helpers:", e)
    sys.exit(1)

# --- PATH CONFIGURATION (THE FIX) ---
# Force output to 'results' folder to align with Gate C
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("figs", exist_ok=True)

# load NGC3198 params
params_path_candidates = [
    "results/all_galaxy_params.json",
    "all_galaxy_params.json"
]
params = None
for p in params_path_candidates:
    if os.path.exists(p):
        params = json.load(open(p))
        print("Loaded params from", p)
        break

if params is None:
    print("Warning: Cannot find all_galaxy_params.json. Using defaults.")
    L, mu, kernel_choice = 50.0, 50.0, "ananta-hybrid"
else:
    ngc = params.get("NGC3198")
    if ngc is None:
        # fallback to first entry
        first = list(params.keys())[0]
        ngc = params[first]
        print("Warning: NGC3198 not found in JSON — using first entry:", first)

    L = float(ngc.get("L"))
    mu = float(ngc.get("mu"))
    kernel_choice = ngc.get("kernel", "ananta-hybrid")

print(f"Using (L, mu) from NGC3198 -> L={L}, mu={mu}, kernel={kernel_choice}")

# define zero-refit targets
targets = ["DDO154", "UGC00128", "F568-3"] 

# load galaxy metadata
gal_table = read_galaxy_table(os.path.join("data","galaxies.csv")) if os.path.exists(os.path.join("data","galaxies.csv")) else None
out_results = []

for name in targets:
    # find metadata
    gal = None
    if gal_table:
        gal = next((g for g in gal_table if g["name"] == name), None)
    if gal is None:
        print(f"[SKIP] metadata for {name} not found.")
        continue

    # predict
    pred = predict_rc_for_params(gal, L, mu, kernel_choice)
    obs = try_read_observed_rc(name)
    if pred is None or obs is None:
        print(f"[SKIP] {name}: missing prediction or observed RC.")
        continue

    R_pred, V_pred, V_b, V_k = pred
    R_obs, V_obs = obs

    # compute MAFE
    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
    m = np.isfinite(Vp) & np.isfinite(V_obs)
    score = float('nan')
    n_points = int(np.sum(m))
    if n_points > 0:
        score = float(np.median(np.abs(Vp[m] - V_obs[m]) / np.clip(V_obs[m], 1e-6, None)))

    print(f"[RESULT] {name}  MAFE = {score:.6e} over {n_points} points")
    out_results.append({"galaxy": name, "mafe": score, "n_points": n_points})

    # Plotting
    try:
        plt.figure(figsize=(6, 4.5))
        plt.plot(R_obs, V_obs, marker="o", linestyle="", color="black", label="Observed", ms=4, alpha=0.7)
        plt.plot(R_pred, V_pred, color="red", label=f"Universal Fit (L={L}, $\\mu$={mu})")
        plt.plot(R_pred, V_b, ":", color="cyan", label="Baryons", linewidth=1)
        plt.xlabel("R [kpc]"); plt.ylabel("V [km/s]")
        plt.title(f"Gate B: {name} (Zero-Refit Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figs/{name}_gateB.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

# save results to the correct folder
outpath = os.path.join(out_dir, "gate_b_results.json")
with open(outpath, "w") as f:
    json.dump(out_results, f, indent=2)
print("Gate B results saved to", outpath)
