#!/usr/bin/env python3
import os
import subprocess
import time

# 1) Path to your target list
TARGET_FILE = "targets_plus20.txt"

# 2) Where to store results
RESULT_DIR = "results/plus20_local"
FIG_DIR = "figs/plus20_local"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 3) Read galaxy names
with open(TARGET_FILE, "r") as f:
    galaxies = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(galaxies)} galaxies:")
for g in galaxies:
    print("  -", g)

print("\nStarting fleet run...\n")

for gal in galaxies:
    print("="*60)
    print(f"Running {gal}")
    print("="*60)

    # command calling your existing runner
    cmd = [
        "python", "run_sparc_lite.py",
        "--galaxy", gal,
        "--L", "50",
        "--mu", "50",
        "--beta", "1.15",
        "--outdir", RESULT_DIR,
        "--figdir", FIG_DIR
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] {gal} FAILED\n")
        with open(os.path.join(RESULT_DIR, "errors.log"), "a") as log:
            log.write(gal + "\n")
    time.sleep(1)  # brief cool-down

print("\n=== Fleet run complete ===\n")
print(f"Results saved in: {RESULT_DIR}")
print(f"Figures saved in: {FIG_DIR}")
