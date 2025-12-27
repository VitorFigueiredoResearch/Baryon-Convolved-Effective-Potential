[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16967259.svg)](https://doi.org/10.5281/zenodo.16967259)

[![CI](https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential/actions/workflows/ci.yml/badge.svg)](https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential/actions/workflows/ci.yml)

# H1 — Baryon-Convolved Single-Field Potential

**Status:** Frozen phenomenological model  
**Domain:** Galaxy rotation curves (SPARC)  
**Author:** Vítor M. F. Figueiredo  
**ORCID:** https://orcid.org/0009-0004-7358-4622  
**Zenodo DOI:** https://doi.org/10.5281/zenodo.16967259

---

## Overview

This repository contains the full numerical pipeline, frozen results, and
manuscript source for **H1**, a phenomenological framework that models
galaxy rotation curves using a **single effective gravitational potential**
constructed from a **nonlocal convolution of the observed baryonic mass
distribution**.

H1 is intentionally **descriptive rather than predictive**.
Its purpose is to test—using real data and a controlled, resolution-stable
numerical pipeline—whether **finite-range, nonlocal baryonic response models**
can reproduce observed rotation-curve morphologies **without invoking
particle dark matter** or modifying local gravitational laws.

The framework is evaluated against **175 disk galaxies** from the
**SPARC (Spitzer Photometry & Accurate Rotation Curves) database**.

All successes and failures are retained as results.

---

## Scientific Scope

H1 is **not**:
- a fundamental theory of gravity  
- a relativistic or cosmological model  
- a replacement for ΛCDM or MOND  

H1 **is**:
- a frozen phenomenological baseline  
- a reproducible numerical experiment  
- a falsifiable testbed for nonlocal baryon–potential coupling  

The model uses **global parameters** shared across the entire galaxy sample.
No per-galaxy tuning is permitted.

---

## Repository Structure

The repository is intentionally minimal and reflects the frozen scope of H1:


Baryon-Convolved-Effective-Potential/
├── paper/            # LaTeX source of the H1 manuscript
├── src/              # Core numerical components (kernels, utilities)
│   └── kernels.py
├── results/          # Frozen per-galaxy CSV / JSON outputs (DX = 1.0)
├── sparc_extractor/  # Data preparation utilities (SPARC ingestion)
├── tools/            # Auxiliary diagnostics (not required for reproduction)
├── run_sparc_lite.py # Main frozen H1 pipeline
├── README.md
├── LICENSE.txt
└── CITATION.cff


Exploratory scripts and tuning gates used during development have been
intentionally removed to preserve a clean, submission-grade interface.

---

## Data Source

This work uses the **SPARC** database.

**Primary data citation:**

> Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).  
> *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves.*  
> The Astronomical Journal, 152(6), 157.  
> https://doi.org/10.3847/0004-6256/152/6/157

Users of this repository **must cite SPARC** in any derivative work.

---

## Installation

### Option A — Conda (recommended)


conda env create -f env/environment.yml
conda activate H1-env


### Option B — pip / venv


python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt


---

## Running the Pipeline

To reproduce the main H1 results:


python run_sparc_lite.py


Outputs are written to:

* `results/` — per-galaxy summaries and diagnostics

The numerical pipeline is **frozen**.
Do not modify kernel definitions, grid resolution, normalisation logic,
or fitting procedures if you intend to reproduce the published results.

Optional verbosity and diagnostic flags are documented in the code header.

---

## Numerical Integrity

The pipeline includes explicit checks for:

* FFT normalisation and zero-mode suppression
* kernel integral consistency
* radial tapering stability
* resolution invariance (Δx = 0.5–1.0 kpc)

These tests are described in **Appendix A** of the manuscript.

---

## Reproducibility Statement

* No galaxy-specific tuning
* No post-hoc parameter adjustment
* No hidden priors or adaptive refits
* Identical pipeline applied to all galaxies

Any modification to the code must be stated explicitly in derivative work.

---

## Citation

If you use this code or its results, please cite:


Figueiredo, V. M. F. (2025).
H1: A Baryon-Convolved Single-Field Potential.
Zenodo. https://doi.org/10.5281/zenodo.16967259

---

## License

This repository is released for **scientific research and reproducibility**.

You are free to:

* inspect the methods
* run the pipeline
* reproduce the results
* build upon the framework

You may not:

* remove attribution
* misrepresent H1 as a predictive or fundamental theory
* imply endorsement beyond the stated scope

---

## Final Note

H1 is designed as a **baseline**.

Whether it ultimately succeeds or fails, its purpose is to establish—clearly,
honestly, and reproducibly—how much of galactic rotation-curve structure can be
captured by **nonlocal baryonic response alone**.
