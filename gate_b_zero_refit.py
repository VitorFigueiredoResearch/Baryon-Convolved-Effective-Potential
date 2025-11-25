# gate_b_zero_refit.py
# Gate B: Zero-Refit Test
# Usage: python gate_b_zero_refit.py
# Expects: repo root with run_sparc_lite.py (imports predict_rc_for_params etc.)
# Writes: /mnt/data/gate_b_results.json

import json, numpy as np, os, sys
from pathlib import Path

# Try to import helper functions from your script
# Make sure run_sparc_lite.py is importable (repo root) or adapt path
try:
    from run_sparc_lite import predict_rc_for_params, read_galaxy_table, try_read_observed_rc
except Exception as e:
    print("ERROR importing run_sparc_lite helpers:", e)
    print("Run this from the repo root or adjust PYTHONPATH. Exiting.")
    sys.exit(1)

# load NGC3198 params (choose the canonical JSON from your CI)
params_path_candidates = [
    "/mnt/data/all_galaxy_params.json",   # CI-uploaded artifact
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
    raise SystemExit("Cannot find all_galaxy_params.json (check paths).")

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

# define zero-refit targets (change list as you wish)
targets = ["DDO154", "UGC00128", "F568-3"]  # ensure these are available in your SPARC data

# load galaxy metadata (this will read data/galaxies.csv if present).
gal_table = read_galaxy_table(os.path.join("data","galaxies.csv")) if os.path.exists(os.path.join("data","galaxies.csv")) else None
out_results = []

for name in targets:
    # find metadata
    gal = None
    if gal_table:
        gal = next((g for g in gal_table if g["name"] == name), None)
    if gal is None:
        print(f"[SKIP] metadata for {name} not found in data/galaxies.csv. Try using NIGHTMARE_FLEET fallback in run_sparc_lite by leaving TARGET_GALAXY=None.")
        # still try fallback: construct minimal param set if present in NIGHTMARE_FLEET style (optional)
        continue

    # predict with exact (L,mu)
    pred = predict_rc_for_params(gal, L, mu, kernel_choice)
    obs = try_read_observed_rc(name)
    if pred is None or obs is None:
        print(f"[SKIP] {name}: missing prediction or observed RC.")
        continue

    R_pred, V_pred, V_b, V_k = pred
    R_obs, V_obs = obs

    # interpolate predicted velocity onto observed radii, compute MAFE (same metric used in main run)
    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
    m = np.isfinite(Vp) & np.isfinite(V_obs)
    score = float('nan')
    n_points = int(np.sum(m))
    if n_points > 0:
        score = float(np.median(np.abs(Vp[m] - V_obs[m]) / np.clip(V_obs[m], 1e-6, None)))

    print(f"[RESULT] {name}  MAFE = {score:.6e} over {n_points} points")
    out_results.append({"galaxy": name, "mafe": score, "n_points": n_points})

# save results
outpath = "/mnt/data/gate_b_results.json"
with open(outpath, "w") as f:
    json.dump(out_results, f, indent=2)
print("Gate B results saved to", outpath)
