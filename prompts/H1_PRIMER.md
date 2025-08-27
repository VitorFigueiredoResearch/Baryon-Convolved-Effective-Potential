PROJECT: H1 — baryon-built effective potential for disk galaxies.

CORE CLAIM. Build a single scalar field Φ_eff from observed baryons via a cored, finite-range convolution. Two global parameters only: length L (kpc) and dimensionless gain μ. Stars follow −∇Φ_eff for rotation curves. Lensing uses the same Φ_eff through Poisson: ρ_eff ≡ ∇²Φ_eff/(4πG) → Σ → ΔΣ; no refit for lensing.

KERNELS. Use cored, positive acceleration kernels
K_Plummer(r; L) = 1/[4π(r² + L²)],  K_exp-core(r; L) = e^{−r/L}/[4π(r² + L²)].
Avoid 1/r tails (do not hard-wire flat asymptotes). Units: [U]=L⁻¹, [K]=L⁻².

NUMERICS. Authoritative route: 3D FFT (potential → gradient). Axisymmetric Hankel is a cross-check only. Truncate kernel at R_trunc = 6L; pad the domain; set k=0 potential mode to zero.

DATA & PRIORS. SPARC subset. M/L priors: Υ_[3.6],disk = 0.5±0.1; Υ_[3.6],bulge = 0.7±0.1 (Gaussian, ±2σ). Gas = 1.33×HI. Global fit of (L, μ) by cross-validation; report MAFE & RMSE. Include a dedicated gas-dwarf fold; give bootstrap CIs.

BASELINE. Per-galaxy Burkert (main) and NFW (sensitivity) under identical baryonic priors.

GATES (falsification). RC residuals (MAFE ≤ 10% & CV-RMSE ≤ baseline+5%); BTFR slope 3.8–4.2 with σ_int ≤ 0.12 dex; RAR scatter ≤ 0.13 dex with no trend (α=0.01); lensing: ≥10% residual-variance reduction at 2σ vs baselines (baryons-only and NFW with Dutton–Macciò).

SINGLE-FIELD STATEMENT. Keep Φ_eff unique for both dynamics and lensing via the Poisson route (no auxiliary fields, no galaxy-specific parameters).

STYLE. Return patch-style LaTeX (delta edits), enforce en dashes (Tully–Fisher, mass–concentration, galaxy–galaxy), Planck-like tone; never recode 1/r kernels; keep Single-Field wording aligned with the Poisson mapping.
