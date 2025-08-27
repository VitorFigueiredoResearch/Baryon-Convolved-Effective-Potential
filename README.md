[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16967259.svg)](https://doi.org/10.5281/zenodo.16967259)

[![CI](https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential/actions/workflows/ci.yml/badge.svg)](https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential/actions/workflows/ci.yml)


# H1 — Baryon‑Convolved Effective Potential (rc)

**What this repo contains**
- **Paper** (`paper/`): LaTeX source of the H1 draft (single‑field potential, global L and μ).
- **Code** (`src/`): 3D‑FFT pipeline skeleton for rotation curves (RC), BTFR, RAR, and lensing prediction.
- **Config** (`config/`): Preregistered thresholds, paths, and kernel choice.
- **Folds** (`folds/`): Deterministic CV split.
- **Prompts** (`prompts/`): Tiny primer/style/patch-mode snippets for fast model re-sync.
- **CI** (`.github/workflows/`): Python experiment run and PDF build on push.

> **Single‑Field statement.** One scalar potential \(\Phi_{\rm eff}\) from baryons is used for both dynamics \((-\nabla\Phi_{\rm eff})\) and lensing (Poisson→Σ→ΔΣ). No per‑galaxy refits for lensing.

## Getting started
```bash
# Python environment (pip)
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt

# Or conda
conda env create -f env/environment.yml && conda activate h1-evolving-potential

# Run full experiment (place data paths in config/h1_config.yaml)
bash ./run_experiment.sh
```

Artifacts are saved to `figs/` and `results/` (uploaded by CI).

## Build the paper
```bash
cd paper
latexmk -pdf main.tex
```

## Preregistered gates (brief)
- **RC:** MAFE ≤ 10% and CV‑RMSE ≤ baseline + 5%  
- **BTFR:** slope 3.8–4.2 with σ_int ≤ 0.12 dex  
- **RAR:** scatter ≤ 0.13 dex with no trend (α=0.01)  
- **Lensing (prediction):** ≥10% residual‑variance reduction at 2σ vs baseline (baryons‑only + NFW with Dutton–Macciò)

