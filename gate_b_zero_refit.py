# gate_b_zero_refit.py
# Gate B: Zero-Refit Test
# Usage: python gate_b_zero_refit.py
# Writes: results/gate_b_results.json and figs/*_gateB.png

import json, numpy as np, os, sys
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Name Normalization Helper
# -------------------------
def normalize_name(s):
    if s is None:
        return ""
    s = s.strip().lower()
    for ch in ["_", " ", "\u2011", "\u2012", "\u2013", "\u2014"]:
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s


# -------------------------
# Import required helpers
# -------------------------
try:
    from run_sparc_lite import predict_rc_for_params, read_galaxy_table, try_read_observed_rc
except Exception as e:
    print("ERROR importing run_sparc_lite helpers:", e)
    sys.exit(1)


# -------------------------
# Output folders
# -------------------------
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("figs", exist_ok=True)


# -------------------------
# Load NGC3198 params
# -------------------------
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
    print("Warning: cannot find all_galaxy_params.json — using defaults.")
    L, mu, kernel_choice = 50.0, 50.0, "ananta-hybrid"
else:
    ngc = params.get("NGC3198")
    if ngc is None:
        first = list(params.keys())[0]
        ngc = params[first]
        print("Warning: NGC3198 not found — using first entry:", first)

    L = float(ngc.get("L"))
    mu = float(ngc.get("mu"))
    kernel_choice = ngc.get("kernel", "ananta-hybrid")

print(f"Using (L, mu) from NGC3198 -> L={L}, mu={mu}, kernel={kernel_choice}")


# -------------------------
# Load galaxy table
# -------------------------
gal_csv = os.path.join("data", "galaxies.csv")
gal_table = None
if os.path.exists(gal_csv):
    gal_table = read_galaxy_table(gal_csv)
    print(f"Loaded galaxy table with {len(gal_table)} entries.")
else:
    print("ERROR: Cannot find data/galaxies.csv")
    sys.exit(1)


# -------------------------
# Build normalized map
# -------------------------
gal_norm_map = {}
for g in gal_table:
    key = normalize_name(g.get("name", ""))
    gal_norm_map.setdefault(key, []).append(g)


# -------------------------
# Galaxies to test (Zero-Refit)
# -------------------------
targets = ["DDO154", "UGC00128", "F568-3"]


# -------------------------
# Main Loop
# -------------------------
out_results = []

for name in targets:
    print("\n--- Processing", name, "---")

    # find metadata via normalized dictionary
    norm_target = normalize_name(name)
    candidates = gal_norm_map.get(norm_target, [])

    if len(candidates) == 1:
        gal = candidates[0]
    elif len(candidates) > 1:
        print(f"Warning: multiple matches for {name}, using first.")
        gal = candidates[0]
    else:
        # weak fuzzy match fallback
        gal = next((g for k, glist in gal_norm_map.items()
                    if norm_target in k or k in norm_target
                    for g in glist), None)

    if gal is None:
        print(f"[SKIP] {name}: metadata not found.")
        continue

    # load observed RC
    obs = try_read_observed_rc(name)
    if obs is None:
        print(f"[SKIP] {name}: observed RC not found.")
        continue
    R_obs, V_obs = obs

    # predicted (zero-refit)
    pred = predict_rc_for_params(gal, L, mu, kernel_choice)
    if pred is None:
        print(f"[SKIP] {name}: prediction failed.")
        continue

    R_pred, V_pred, V_b, V_k = pred

    # MAFE
    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
    m = np.isfinite(Vp) & np.isfinite(V_obs)
    if np.sum(m) == 0:
        score = float('nan')
        n_points = 0
    else:
        score = float(np.median(np.abs(Vp[m] - V_obs[m]) / np.clip(V_obs[m], 1e-6, None)))
        n_points = int(np.sum(m))

    print(f"[RESULT] {name}: MAFE = {score:.5f} over {n_points} points")
    out_results.append({"galaxy": name, "mafe": score, "n_points": n_points})

    # Plot
    try:
        plt.figure(figsize=(6, 4.5))
        plt.plot(R_obs, V_obs, "o", color="black", markersize=4, label="Observed")
        plt.plot(R_pred, V_pred, "-", color="orange", label="H1 Total")
        plt.plot(R_pred, V_b, ":", color="cyan", label="Baryons")
        plt.xlabel("R [kpc]"); plt.ylabel("V [km/s]")
        plt.title(f"Gate B: {name} (Zero-Refit)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figs/{name}_gateB.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)


# -------------------------
# Save results
# -------------------------
outpath = os.path.join(out_dir, "gate_b_results.json")
with open(outpath, "w") as f:
    json.dump(out_results, f, indent=2)

print("\nGate B results saved to", outpath)
