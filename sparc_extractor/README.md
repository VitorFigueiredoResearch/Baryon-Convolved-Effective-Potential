# SPARC Galaxy Parameter Extraction for H1

This directory contains a self-contained, reproducible pipeline that generates
the baryonic parameter table used in the H1 Baryon-Convolved Potential Model.

## Contents

- extract_sparc_params_v5.py — main extraction script
- diagnose_table1.py — validator for SPARC Table1 alignment
- raw/ — unmodified SPARC data downloaded from CDS:
    * table1.mrt (Galaxy Sample, Lelli+2016)
    * Bulges.mrt (Bulge luminosities)
    * wise_ii_table1.mrt (stellar mass estimates)
    * rotmod/ (HI rotation curve files)
- output/ — machine-generated products:
    * galaxies.csv
    * galaxies_h1_lines.json
    * missing_report.txt
    * batches/

## How to run

cd sparc_extractor
python extract_sparc_params_v5.py


This regenerates the H1 galaxy dataset exactly.

## Citation

If you use this pipeline, please cite:
- Lelli, McGaugh & Schombert (2016), SPARC database
## Citations

If you use the H1 model or the SPARC-based galaxy parameters, please cite:

**SPARC (Lelli, McGaugh & Schombert 2016)**  
Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).  
*SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves.*  
The Astronomical Journal, 152, 157.  
doi: 10.3847/0004-6256/152/6/157

**H1 Model (Figueiredo 2025)**  
Figueiredo, V. M. F. (2025). *Baryon-Convolved Effective Potential (H1).*  
Zenodo. doi: 10.5281/zenodo.16967259  
Repository: https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential

