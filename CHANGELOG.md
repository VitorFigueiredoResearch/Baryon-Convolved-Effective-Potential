# Changelog

## Version 2.2.0 (March 2026)

### Added
- `comparative_analysis/comparative_validation/` — H2 tier audit (30 explicit Tier A+B; 44 Tier C no-archive), RAR Spearman correlation result (ρ=+0.049, null), and full 74-galaxy H2 explicit summary CSV
- `comparative_analysis/final_validation/` — pre-submission go/no-go memo, exclusion consistency check, metric harmonization check, MOND/RAR defensibility audit
- `comparative_analysis/mond/` and `nfw/` — perturbation summary CSVs (888-row RAR; NFW long-format), diagnostic and narrative reports
- `comparative_analysis/metric_harmonization/` — harmonized metric CSVs and inspection reports
- `scripts/Figures/generate_three_way_comparison.py` v2.0 — three-way comparison figure (NFW/RAR/H2); H2 uses 30 explicit archived galaxies only; 44 Tier C not plotted; both Spearman results annotated

### Changed
- Manuscript target: resubmitted to Monthly Notices of the Royal Astronomical Society (MNRAS) as comparative paper
- Paper title: *"Inner-Region Scatter Response to Bounded Perturbations: A Comparative Analysis of 74 SPARC Galaxies"*
- H2 median |Δσ| updated to 0.000860 dex based on 30 explicit Tier A+B archived galaxies (was 0.0005 dex from earlier 9-galaxy estimate)
- README updated to reflect comparative-paper state and correct H2 archive coverage

   
   ## v1.3 (March 2026)
   - Added scripts/figures/ folder with acceleration-space plot generation
   - Updated manuscript to ApJ submission version
   - Paper now in peer review at ApJ (Manuscript #AAS74932)
   
   ## v1.2 (Initial release)
   - Complete H2 validation pipeline
   - 80-galaxy SPARC sample
