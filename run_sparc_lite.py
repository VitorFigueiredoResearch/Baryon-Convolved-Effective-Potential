# ✦        ·        ✧           ·        ✦  ·        ✧           ·                     ✧ 
#      ·        ✦        ·             ✧      ✦        .         .         ✦        .         
#  ✧       @       H1 — Baryon-Convolved Single-Field Potential             @         .   
#      ·        ✦        ·             ✧
# ✦        ·        ✧           ·        ✦          .         .           *             ✦
#
# Author: Vítor M. F. Figueiredo
# ORCID:  https://orcid.org/0009-0004-7358-4622
# Years:  2024–2025
#
# This code implements the frozen numerical pipeline used in:
#   "H1: A Baryon-Convolved Single-Field Potential"
#
# Repository / DOI:
#   https://github.com/VitorFigueiredoResearch/Baryon-Convolved-Effective-Potential
#
# Built with care for numerical integrity,
# empirical honesty, and reproducibility.
# ✦        ·        ✧           ·        ✦         ✧           @               . 

# Notes:
#  - Set TARGET_GALAXY = "NGC3198" to debug one galaxy
#  - Set TARGET_GALAXY = None to run the whole fleet

import os
# ---- LOGGING / VERBOSITY CONTROL ----
VERBOSE = os.environ.get("H1_VERBOSE", "0") == "1"
DEBUG   = os.environ.get("H1_DEBUG",   "0") == "1"
SHOW_FIX = os.environ.get("H1_FIX", "1") == "1"  # default ON

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def fix(*args, **kwargs):
    if SHOW_FIX:
        print(*args, **kwargs)
def ok(*args, **kwargs):
    print(*args, **kwargs)

import csv
import json
import gc
import urllib.request
import zipfile
import re
import numpy as np
import matplotlib.pyplot as plt

# Local physics modules 
from src.kernels import U_plummer, U_exp_core, U_ananta_hybrid
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G

# ---- CONFIG ----
RADIAL_BINS = 30

# Kernel search lists 
KERNELS = ("ananta-hybrid",)
L_LIST = [10.0, 30.0, 50.0, 80.0, 120.0, 200.0]
MU_LIST = [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]

# Target: set to a string to run ONE galaxy (fast debug), or None for all from CSV
TARGET_GALAXY = "NGC4559"   # e.g. "NGC3198" for fast debug

# If you need to flip kernel sign for testing (polarity fix), change to -1.0
POLARITY_SIGN = 1.0

# ---- Backup fleet (fallback if CSV not present or empty) ----
NIGHTMARE_FLEET = [
    # --- The Original SPARC full Fleet ---
{"name": "CamB", "Rd_star": 0.47, "Mstar": 37500000.0, "hz_star": 0.09, "Rd_gas": 0.85, "Mgas": 15960000.000000002, "hz_gas": 0.15},
{"name": "D512-2", "Rd_star": 1.24, "Mstar": 162500000.0, "hz_star": 0.25, "Rd_gas": 2.23, "Mgas": 107730000.0, "hz_gas": 0.15},
{"name": "D564-8", "Rd_star": 0.61, "Mstar": 16500000.0, "hz_star": 0.12, "Rd_gas": 1.1, "Mgas": 38570000.00000001, "hz_gas": 0.15},
{"name": "D631-7", "Rd_star": 0.7, "Mstar": 98000000.0, "hz_star": 0.14, "Rd_gas": 1.26, "Mgas": 385700000.0, "hz_gas": 0.15},
{"name": "DDO064", "Rd_star": 0.69, "Mstar": 78500000.0, "hz_star": 0.14, "Rd_gas": 1.24, "Mgas": 280630000.0, "hz_gas": 0.15},
{"name": "DDO154", "Rd_star": 0.37, "Mstar": 26500000.0, "hz_star": 0.07, "Rd_gas": 0.67, "Mgas": 365750000.00000006, "hz_gas": 0.15},
{"name": "DDO161", "Rd_star": 1.22, "Mstar": 274000000.0, "hz_star": 0.24, "Rd_gas": 2.2, "Mgas": 1832740000.0, "hz_gas": 0.15},
{"name": "DDO168", "Rd_star": 1.02, "Mstar": 95500000.0, "hz_star": 0.2, "Rd_gas": 1.84, "Mgas": 549290000.0, "hz_gas": 0.15},
{"name": "DDO170", "Rd_star": 1.95, "Mstar": 271500000.0, "hz_star": 0.39, "Rd_gas": 3.51, "Mgas": 977550000.0, "hz_gas": 0.15},
{"name": "ESO079-G014", "Rd_star": 5.08, "Mstar": 25866500000.0, "hz_star": 1.02, "Rd_gas": 9.14, "Mgas": 4176200000.0000005, "hz_gas": 0.15},
{"name": "ESO116-G012", "Rd_star": 1.51, "Mstar": 2146000000.0, "hz_star": 0.3, "Rd_gas": 2.72, "Mgas": 1440390000.0, "hz_gas": 0.15},
{"name": "ESO444-G084", "Rd_star": 0.46, "Mstar": 35500000.0, "hz_star": 0.09, "Rd_gas": 0.83, "Mgas": 179550000.00000003, "hz_gas": 0.15},
{"name": "ESO563-G021", "Rd_star": 5.45, "Mstar": 155588500000.0, "hz_star": 1.09, "Rd_gas": 9.81, "Mgas": 32316339999.999996, "hz_gas": 0.15},
{"name": "F561-1", "Rd_star": 2.79, "Mstar": 2038500000.0, "hz_star": 0.56, "Rd_gas": 5.02, "Mgas": 2157260000.0000005, "hz_gas": 0.15},
{"name": "F563-1", "Rd_star": 3.52, "Mstar": 951500000.0, "hz_star": 0.7, "Rd_gas": 6.34, "Mgas": 4256000000.0, "hz_gas": 0.15},
{"name": "F563-V1", "Rd_star": 3.79, "Mstar": 770000000.0, "hz_star": 0.76, "Rd_gas": 6.82, "Mgas": 811300000.0, "hz_gas": 0.15},
{"name": "F563-V2", "Rd_star": 2.43, "Mstar": 1493000000.0, "hz_star": 0.49, "Rd_gas": 4.37, "Mgas": 2884770000.0, "hz_gas": 0.15},
{"name": "F565-V2", "Rd_star": 2.17, "Mstar": 279500000.0, "hz_star": 0.43, "Rd_gas": 3.91, "Mgas": 929670000.0, "hz_gas": 0.15},
{"name": "F567-2", "Rd_star": 3.08, "Mstar": 1067000000.0, "hz_star": 0.62, "Rd_gas": 5.54, "Mgas": 3257170000.0, "hz_gas": 0.15},
{"name": "F568-1", "Rd_star": 5.18, "Mstar": 3126000000.0, "hz_star": 1.04, "Rd_gas": 9.32, "Mgas": 5982340000.000001, "hz_gas": 0.15},
{"name": "F568-3", "Rd_star": 4.99, "Mstar": 4173000000.0, "hz_star": 1.0, "Rd_gas": 8.98, "Mgas": 4249349999.9999995, "hz_gas": 0.15},
{"name": "F568-V1", "Rd_star": 2.85, "Mstar": 1912500000.0, "hz_star": 0.57, "Rd_gas": 5.13, "Mgas": 3313030000.0000005, "hz_gas": 0.15},
{"name": "F571-8", "Rd_star": 3.56, "Mstar": 5082000000.0, "hz_star": 0.71, "Rd_gas": 6.41, "Mgas": 2370060000.0, "hz_gas": 0.15},
{"name": "F571-V1", "Rd_star": 2.47, "Mstar": 924500000.0, "hz_star": 0.49, "Rd_gas": 4.45, "Mgas": 1618610000.0, "hz_gas": 0.15},
{"name": "F574-1", "Rd_star": 4.46, "Mstar": 3268500000.0, "hz_star": 0.89, "Rd_gas": 8.03, "Mgas": 4686920000.000001, "hz_gas": 0.15},
{"name": "F574-2", "Rd_star": 3.76, "Mstar": 1438500000.0, "hz_star": 0.75, "Rd_gas": 6.77, "Mgas": 2262330000.0000005, "hz_gas": 0.15},
{"name": "F579-V1", "Rd_star": 3.37, "Mstar": 5924000000.0, "hz_star": 0.67, "Rd_gas": 6.07, "Mgas": 2985850000.0, "hz_gas": 0.15},
{"name": "F583-1", "Rd_star": 2.36, "Mstar": 493000000.0, "hz_star": 0.47, "Rd_gas": 4.25, "Mgas": 2827580000.0, "hz_gas": 0.15},
{"name": "F583-4", "Rd_star": 1.93, "Mstar": 857500000.0, "hz_star": 0.39, "Rd_gas": 3.47, "Mgas": 852530000.0, "hz_gas": 0.15},
{"name": "IC2574", "Rd_star": 2.78, "Mstar": 508000000.0, "hz_star": 0.56, "Rd_gas": 5.0, "Mgas": 1377880000.0000002, "hz_gas": 0.15},
{"name": "IC4202", "Rd_star": 4.78, "Mstar": 89874500000.0, "hz_star": 0.96, "Rd_gas": 8.6, "Mgas": 16393580000.0, "hz_gas": 0.15},
{"name": "KK98-251", "Rd_star": 1.34, "Mstar": 42500000.0, "hz_star": 0.27, "Rd_gas": 2.41, "Mgas": 152950000.0, "hz_gas": 0.15},
{"name": "NGC0024", "Rd_star": 1.34, "Mstar": 1944500000.0, "hz_star": 0.27, "Rd_gas": 2.41, "Mgas": 899080000.0000001, "hz_gas": 0.15},
{"name": "NGC0055", "Rd_star": 6.11, "Mstar": 2314000000.0, "hz_star": 1.22, "Rd_gas": 11.0, "Mgas": 2081450000.0000002, "hz_gas": 0.15},
{"name": "NGC0100", "Rd_star": 1.66, "Mstar": 1616000000.0, "hz_star": 0.33, "Rd_gas": 2.99, "Mgas": 2646700000.0, "hz_gas": 0.15},
{"name": "NGC0247", "Rd_star": 3.74, "Mstar": 3666000000.0, "hz_star": 0.75, "Rd_gas": 6.73, "Mgas": 2322180000.0, "hz_gas": 0.15},
{"name": "NGC0289", "Rd_star": 6.74, "Mstar": 36032500000.0, "hz_star": 1.35, "Rd_gas": 12.13, "Mgas": 36533770000.00001, "hz_gas": 0.15},
{"name": "NGC0300", "Rd_star": 1.75, "Mstar": 1461000000.0, "hz_star": 0.35, "Rd_gas": 3.15, "Mgas": 1244880000.0000002, "hz_gas": 0.15},
{"name": "NGC0801", "Rd_star": 8.72, "Mstar": 156285000000.0, "hz_star": 1.74, "Rd_gas": 15.7, "Mgas": 30857330000.0, "hz_gas": 0.15},
{"name": "NGC0891", "Rd_star": 2.55, "Mstar": 69170000000.0, "hz_star": 0.51, "Rd_gas": 4.59, "Mgas": 5934460000.0, "hz_gas": 0.15},
{"name": "NGC1003", "Rd_star": 1.61, "Mstar": 3410000000.0, "hz_star": 0.32, "Rd_gas": 2.9, "Mgas": 7820400000.0, "hz_gas": 0.15},
{"name": "NGC1090", "Rd_star": 3.53, "Mstar": 36022500000.0, "hz_star": 0.71, "Rd_gas": 6.35, "Mgas": 11681390000.0, "hz_gas": 0.15},
{"name": "NGC1705", "Rd_star": 0.39, "Mstar": 266500000.0, "hz_star": 0.08, "Rd_gas": 0.7, "Mgas": 184870000.00000003, "hz_gas": 0.15},
{"name": "NGC2366", "Rd_star": 0.65, "Mstar": 118000000.0, "hz_star": 0.13, "Rd_gas": 1.17, "Mgas": 860510000.0000001, "hz_gas": 0.15},
{"name": "NGC2403", "Rd_star": 1.39, "Mstar": 5020500000.0, "hz_star": 0.28, "Rd_gas": 2.5, "Mgas": 4254670000.0, "hz_gas": 0.15},
{"name": "NGC2683", "Rd_star": 2.18, "Mstar": 40207500000.0, "hz_star": 0.44, "Rd_gas": 3.92, "Mgas": 1869980000.0, "hz_gas": 0.15},
{"name": "NGC2841", "Rd_star": 3.64, "Mstar": 94060500000.0, "hz_star": 0.73, "Rd_gas": 6.55, "Mgas": 13000750000.000002, "hz_gas": 0.15},
{"name": "NGC2903", "Rd_star": 2.33, "Mstar": 40931500000.0, "hz_star": 0.47, "Rd_gas": 4.19, "Mgas": 3394160000.0000005, "hz_gas": 0.15},
{"name": "NGC2915", "Rd_star": 0.55, "Mstar": 320500000.0, "hz_star": 0.11, "Rd_gas": 0.99, "Mgas": 675640000.0, "hz_gas": 0.15},
{"name": "NGC2955", "Rd_star": 18.76, "Mstar": 159711000000.0, "hz_star": 3.75, "Rd_gas": 33.77, "Mgas": 38502170000.00001, "hz_gas": 0.15},
{"name": "NGC2976", "Rd_star": 1.01, "Mstar": 1685500000.0, "hz_star": 0.2, "Rd_gas": 1.82, "Mgas": 228760000.0, "hz_gas": 0.15},
{"name": "NGC2998", "Rd_star": 6.2, "Mstar": 75451000000.0, "hz_star": 1.24, "Rd_gas": 11.16, "Mgas": 31189830000.0, "hz_gas": 0.15},
{"name": "NGC3109", "Rd_star": 1.56, "Mstar": 97000000.0, "hz_star": 0.31, "Rd_gas": 2.81, "Mgas": 634410000.0, "hz_gas": 0.15},
{"name": "NGC3198", "Rd_star": 3.14, "Mstar": 19139500000.0, "hz_star": 0.63, "Rd_gas": 5.65, "Mgas": 14455770000.000002, "hz_gas": 0.15},
{"name": "NGC3521", "Rd_star": 2.4, "Mstar": 42418000000.0, "hz_star": 0.48, "Rd_gas": 4.32, "Mgas": 5524820000.0, "hz_gas": 0.15},
{"name": "NGC3726", "Rd_star": 3.4, "Mstar": 35117000000.0, "hz_star": 0.68, "Rd_gas": 6.12, "Mgas": 8609090000.0, "hz_gas": 0.15},
{"name": "NGC3741", "Rd_star": 0.2, "Mstar": 14000000.0, "hz_star": 0.04, "Rd_gas": 0.36, "Mgas": 242060000.0, "hz_gas": 0.15},
{"name": "NGC3769", "Rd_star": 3.38, "Mstar": 9339500000.0, "hz_star": 0.68, "Rd_gas": 6.08, "Mgas": 7353570000.0, "hz_gas": 0.15},
{"name": "NGC3877", "Rd_star": 2.53, "Mstar": 36267500000.0, "hz_star": 0.51, "Rd_gas": 4.55, "Mgas": 1972390000.0000002, "hz_gas": 0.15},
{"name": "NGC3893", "Rd_star": 2.38, "Mstar": 29262500000.0, "hz_star": 0.48, "Rd_gas": 4.28, "Mgas": 7712670000.000001, "hz_gas": 0.15},
{"name": "NGC3917", "Rd_star": 2.63, "Mstar": 10983000000.0, "hz_star": 0.53, "Rd_gas": 4.73, "Mgas": 2511040000.0, "hz_gas": 0.15},
{"name": "NGC3949", "Rd_star": 3.59, "Mstar": 19033500000.0, "hz_star": 0.72, "Rd_gas": 6.46, "Mgas": 4483430000.0, "hz_gas": 0.15},
{"name": "NGC3953", "Rd_star": 4.89, "Mstar": 70650500000.0, "hz_star": 0.98, "Rd_gas": 8.8, "Mgas": 3766560000.0, "hz_gas": 0.15},
{"name": "NGC3972", "Rd_star": 2.18, "Mstar": 7176500000.0, "hz_star": 0.44, "Rd_gas": 3.92, "Mgas": 1614620000.0, "hz_gas": 0.15},
{"name": "NGC3992", "Rd_star": 4.96, "Mstar": 113466000000.0, "hz_star": 0.99, "Rd_gas": 8.93, "Mgas": 22076670000.0, "hz_gas": 0.15},
{"name": "NGC4010", "Rd_star": 2.81, "Mstar": 8596500000.0, "hz_star": 0.56, "Rd_gas": 5.06, "Mgas": 3766560000.0, "hz_gas": 0.15},
{"name": "NGC4013", "Rd_star": 3.53, "Mstar": 39547000000.0, "hz_star": 0.71, "Rd_gas": 6.35, "Mgas": 3946110000.0000005, "hz_gas": 0.15},
{"name": "NGC4051", "Rd_star": 4.65, "Mstar": 47634000000.0, "hz_star": 0.93, "Rd_gas": 8.37, "Mgas": 3587010000.0000005, "hz_gas": 0.15},
{"name": "NGC4068", "Rd_star": 0.59, "Mstar": 118000000.0, "hz_star": 0.12, "Rd_gas": 1.06, "Mgas": 204820000.0, "hz_gas": 0.15},
{"name": "NGC4085", "Rd_star": 1.65, "Mstar": 10862000000.0, "hz_star": 0.33, "Rd_gas": 2.97, "Mgas": 1794170000.0, "hz_gas": 0.15},
{"name": "NGC4088", "Rd_star": 2.58, "Mstar": 53643000000.0, "hz_star": 0.52, "Rd_gas": 4.64, "Mgas": 10940580000.000002, "hz_gas": 0.15},
{"name": "NGC4100", "Rd_star": 2.15, "Mstar": 29697000000.0, "hz_star": 0.43, "Rd_gas": 3.87, "Mgas": 4125660000.0, "hz_gas": 0.15},
{"name": "NGC4138", "Rd_star": 1.51, "Mstar": 22055500000.0, "hz_star": 0.3, "Rd_gas": 2.72, "Mgas": 1972390000.0000002, "hz_gas": 0.15},
{"name": "NGC4157", "Rd_star": 2.32, "Mstar": 52810000000.0, "hz_star": 0.46, "Rd_gas": 4.18, "Mgas": 10940580000.000002, "hz_gas": 0.15},
{"name": "NGC4183", "Rd_star": 2.79, "Mstar": 5419000000.0, "hz_star": 0.56, "Rd_gas": 5.02, "Mgas": 4662980000.0, "hz_gas": 0.15},
{"name": "NGC4214", "Rd_star": 0.51, "Mstar": 570500000.0, "hz_star": 0.1, "Rd_gas": 0.92, "Mgas": 646380000.0000001, "hz_gas": 0.15},
{"name": "NGC4217", "Rd_star": 2.94, "Mstar": 42649500000.0, "hz_star": 0.59, "Rd_gas": 5.29, "Mgas": 3407460000.0, "hz_gas": 0.15},
{"name": "NGC4389", "Rd_star": 2.79, "Mstar": 10664000000.0, "hz_star": 0.56, "Rd_gas": 5.02, "Mgas": 716870000.0000001, "hz_gas": 0.15},
{"name": "NGC4559", "Rd_star": 2.1, "Mstar": 9688500000.0, "hz_star": 0.42, "Rd_gas": 3.78, "Mgas": 7728630000.000001, "hz_gas": 0.15},
{"name": "NGC5005", "Rd_star": 9.45, "Mstar": 89360000000.0, "hz_star": 1.89, "Rd_gas": 17.01, "Mgas": 1702400000.0000002, "hz_gas": 0.15},
{"name": "NGC5033", "Rd_star": 5.16, "Mstar": 55254500000.0, "hz_star": 1.03, "Rd_gas": 9.29, "Mgas": 15047620000.0, "hz_gas": 0.15},
{"name": "NGC5055", "Rd_star": 3.2, "Mstar": 76461000000.0, "hz_star": 0.64, "Rd_gas": 5.76, "Mgas": 15590260000.0, "hz_gas": 0.15},
{"name": "NGC5371", "Rd_star": 7.44, "Mstar": 170196500000.0, "hz_star": 1.49, "Rd_gas": 13.39, "Mgas": 14869400000.0, "hz_gas": 0.15},
{"name": "NGC5585", "Rd_star": 1.53, "Mstar": 1471500000.0, "hz_star": 0.31, "Rd_gas": 2.75, "Mgas": 2238390000.0000005, "hz_gas": 0.15},
{"name": "NGC5907", "Rd_star": 5.34, "Mstar": 87712500000.0, "hz_star": 1.07, "Rd_gas": 9.61, "Mgas": 27963250000.0, "hz_gas": 0.15},
{"name": "NGC5985", "Rd_star": 7.01, "Mstar": 104364000000.0, "hz_star": 1.4, "Rd_gas": 12.62, "Mgas": 15409380000.0, "hz_gas": 0.15},
{"name": "NGC6015", "Rd_star": 2.3, "Mstar": 16064499999.999998, "hz_star": 0.46, "Rd_gas": 4.14, "Mgas": 7759220000.0, "hz_gas": 0.15},
{"name": "NGC6195", "Rd_star": 13.94, "Mstar": 195538000000.0, "hz_star": 2.79, "Rd_gas": 25.09, "Mgas": 27806310000.0, "hz_gas": 0.15},
{"name": "NGC6503", "Rd_star": 2.16, "Mstar": 6422500000.0, "hz_star": 0.43, "Rd_gas": 3.89, "Mgas": 2319520000.0000005, "hz_gas": 0.15},
{"name": "NGC6674", "Rd_star": 6.04, "Mstar": 107327000000.0, "hz_star": 1.21, "Rd_gas": 10.87, "Mgas": 42779450000.00001, "hz_gas": 0.15},
{"name": "NGC6789", "Rd_star": 0.31, "Mstar": 50000000.0, "hz_star": 0.06, "Rd_gas": 0.56, "Mgas": 22610000.0, "hz_gas": 0.15},
{"name": "NGC6946", "Rd_star": 2.44, "Mstar": 33086500000.0, "hz_star": 0.49, "Rd_gas": 4.39, "Mgas": 7541100000.0, "hz_gas": 0.15},
{"name": "NGC7331", "Rd_star": 5.02, "Mstar": 125315500000.0, "hz_star": 1.0, "Rd_gas": 9.04, "Mgas": 14719110000.0, "hz_gas": 0.15},
{"name": "NGC7793", "Rd_star": 1.21, "Mstar": 3525000000.0, "hz_star": 0.24, "Rd_gas": 2.18, "Mgas": 1145130000.0, "hz_gas": 0.15},
{"name": "NGC7814", "Rd_star": 2.54, "Mstar": 37264500000.0, "hz_star": 0.51, "Rd_gas": 4.57, "Mgas": 1423100000.0000002, "hz_gas": 0.15},
{"name": "PGC51017", "Rd_star": 0.53, "Mstar": 77500000.0, "hz_star": 0.11, "Rd_gas": 0.95, "Mgas": 267330000.0, "hz_gas": 0.15},
{"name": "UGC00128", "Rd_star": 5.95, "Mstar": 6010000000.0, "hz_star": 1.19, "Rd_gas": 10.71, "Mgas": 9883230000.000002, "hz_gas": 0.15},
{"name": "UGC00191", "Rd_star": 1.58, "Mstar": 1002000000.0, "hz_star": 0.32, "Rd_gas": 2.84, "Mgas": 1786190000.0000002, "hz_gas": 0.15},
{"name": "UGC00634", "Rd_star": 2.45, "Mstar": 1494500000.0, "hz_star": 0.49, "Rd_gas": 4.41, "Mgas": 4871790000.0, "hz_gas": 0.15},
{"name": "UGC00731", "Rd_star": 2.3, "Mstar": 161500000.0, "hz_star": 0.46, "Rd_gas": 4.14, "Mgas": 2403310000.0, "hz_gas": 0.15},
{"name": "UGC00891", "Rd_star": 1.43, "Mstar": 187000000.0, "hz_star": 0.29, "Rd_gas": 2.57, "Mgas": 569240000.0, "hz_gas": 0.15},
{"name": "UGC01230", "Rd_star": 4.34, "Mstar": 3810000000.0, "hz_star": 0.87, "Rd_gas": 7.81, "Mgas": 8551900000.0, "hz_gas": 0.15},
{"name": "UGC01281", "Rd_star": 1.63, "Mstar": 176500000.0, "hz_star": 0.33, "Rd_gas": 2.93, "Mgas": 391020000.0, "hz_gas": 0.15},
{"name": "UGC02023", "Rd_star": 1.55, "Mstar": 654000000.0, "hz_star": 0.31, "Rd_gas": 2.79, "Mgas": 634410000.0, "hz_gas": 0.15},
{"name": "UGC02259", "Rd_star": 1.62, "Mstar": 862500000.0, "hz_star": 0.32, "Rd_gas": 2.92, "Mgas": 657020000.0, "hz_gas": 0.15},
{"name": "UGC02455", "Rd_star": 0.99, "Mstar": 1824500000.0, "hz_star": 0.2, "Rd_gas": 1.78, "Mgas": 1067990000.0000002, "hz_gas": 0.15},
{"name": "UGC02487", "Rd_star": 7.89, "Mstar": 244977500000.0, "hz_star": 1.58, "Rd_gas": 14.2, "Mgas": 23890790000.000004, "hz_gas": 0.15},
{"name": "UGC02885", "Rd_star": 11.4, "Mstar": 201762500000.0, "hz_star": 2.28, "Rd_gas": 20.52, "Mgas": 53299750000.00001, "hz_gas": 0.15},
{"name": "UGC02916", "Rd_star": 6.15, "Mstar": 62076500000.0, "hz_star": 1.23, "Rd_gas": 11.07, "Mgas": 30953090000.0, "hz_gas": 0.15},
{"name": "UGC02953", "Rd_star": 3.55, "Mstar": 129758999999.99998, "hz_star": 0.71, "Rd_gas": 6.39, "Mgas": 10211740000.0, "hz_gas": 0.15},
{"name": "UGC03205", "Rd_star": 3.19, "Mstar": 56821000000.0, "hz_star": 0.64, "Rd_gas": 5.74, "Mgas": 12870410000.0, "hz_gas": 0.15},
{"name": "UGC03546", "Rd_star": 3.79, "Mstar": 50668000000.0, "hz_star": 0.76, "Rd_gas": 6.82, "Mgas": 3557750000.0, "hz_gas": 0.15},
{"name": "UGC03580", "Rd_star": 2.43, "Mstar": 6633000000.0, "hz_star": 0.49, "Rd_gas": 4.37, "Mgas": 5812100000.0, "hz_gas": 0.15},
{"name": "UGC04278", "Rd_star": 2.21, "Mstar": 653500000.0, "hz_star": 0.44, "Rd_gas": 3.98, "Mgas": 1484280000.0000002, "hz_gas": 0.15},
{"name": "UGC04305", "Rd_star": 1.16, "Mstar": 368000000.0, "hz_star": 0.23, "Rd_gas": 2.09, "Mgas": 917700000.0, "hz_gas": 0.15},
{"name": "UGC04325", "Rd_star": 1.86, "Mstar": 1012999999.9999999, "hz_star": 0.37, "Rd_gas": 3.35, "Mgas": 901740000.0000001, "hz_gas": 0.15},
{"name": "UGC04483", "Rd_star": 0.18, "Mstar": 6500000.0, "hz_star": 0.04, "Rd_gas": 0.32, "Mgas": 42560000.0, "hz_gas": 0.15},
{"name": "UGC04499", "Rd_star": 1.73, "Mstar": 776000000.0, "hz_star": 0.35, "Rd_gas": 3.11, "Mgas": 1463000000.0000002, "hz_gas": 0.15},
{"name": "UGC05005", "Rd_star": 3.2, "Mstar": 2049999999.9999998, "hz_star": 0.64, "Rd_gas": 5.76, "Mgas": 4113690000.0, "hz_gas": 0.15},
{"name": "UGC05253", "Rd_star": 8.07, "Mstar": 85791000000.0, "hz_star": 1.61, "Rd_gas": 14.53, "Mgas": 21806680000.000004, "hz_gas": 0.15},
{"name": "UGC05414", "Rd_star": 1.47, "Mstar": 561500000.0, "hz_star": 0.29, "Rd_gas": 2.65, "Mgas": 763420000.0, "hz_gas": 0.15},
{"name": "UGC05716", "Rd_star": 1.14, "Mstar": 294000000.0, "hz_star": 0.23, "Rd_gas": 2.05, "Mgas": 1455020000.0000002, "hz_gas": 0.15},
{"name": "UGC05721", "Rd_star": 0.38, "Mstar": 265500000.0, "hz_star": 0.08, "Rd_gas": 0.68, "Mgas": 747460000.0000001, "hz_gas": 0.15},
{"name": "UGC05750", "Rd_star": 3.46, "Mstar": 1668000000.0, "hz_star": 0.69, "Rd_gas": 6.23, "Mgas": 1461670000.0, "hz_gas": 0.15},
{"name": "UGC05764", "Rd_star": 1.17, "Mstar": 42500000.0, "hz_star": 0.23, "Rd_gas": 2.11, "Mgas": 216790000.0, "hz_gas": 0.15},
{"name": "UGC05829", "Rd_star": 1.99, "Mstar": 282000000.0, "hz_star": 0.4, "Rd_gas": 3.58, "Mgas": 1360590000.0, "hz_gas": 0.15},
{"name": "UGC05918", "Rd_star": 1.66, "Mstar": 116500000.0, "hz_star": 0.33, "Rd_gas": 2.99, "Mgas": 395010000.0, "hz_gas": 0.15},
{"name": "UGC05986", "Rd_star": 1.67, "Mstar": 2347500000.0, "hz_star": 0.33, "Rd_gas": 3.01, "Mgas": 3547110000.0, "hz_gas": 0.15},
{"name": "UGC05999", "Rd_star": 3.22, "Mstar": 1692000000.0, "hz_star": 0.64, "Rd_gas": 5.8, "Mgas": 2689260000.0, "hz_gas": 0.15},
{"name": "UGC06399", "Rd_star": 2.05, "Mstar": 1148000000.0, "hz_star": 0.41, "Rd_gas": 3.69, "Mgas": 896420000.0000001, "hz_gas": 0.15},
{"name": "UGC06446", "Rd_star": 1.49, "Mstar": 494000000.0, "hz_star": 0.3, "Rd_gas": 2.68, "Mgas": 1834070000.0, "hz_gas": 0.15},
{"name": "UGC06614", "Rd_star": 5.1, "Mstar": 62175000000.0, "hz_star": 1.02, "Rd_gas": 9.18, "Mgas": 29111040000.000004, "hz_gas": 0.15},
{"name": "UGC06628", "Rd_star": 2.82, "Mstar": 1869500000.0, "hz_star": 0.56, "Rd_gas": 5.08, "Mgas": 1995000000.0, "hz_gas": 0.15},
{"name": "UGC06667", "Rd_star": 5.15, "Mstar": 698500000.0, "hz_star": 1.03, "Rd_gas": 9.27, "Mgas": 1075970000.0, "hz_gas": 0.15},
{"name": "UGC06786", "Rd_star": 3.6, "Mstar": 36703500000.0, "hz_star": 0.72, "Rd_gas": 6.48, "Mgas": 6689900000.000001, "hz_gas": 0.15},
{"name": "UGC06787", "Rd_star": 5.37, "Mstar": 49128000000.0, "hz_star": 1.07, "Rd_gas": 9.67, "Mgas": 6689900000.000001, "hz_gas": 0.15},
{"name": "UGC06818", "Rd_star": 1.39, "Mstar": 794000000.0, "hz_star": 0.28, "Rd_gas": 2.5, "Mgas": 1435070000.0, "hz_gas": 0.15},
{"name": "UGC06917", "Rd_star": 2.76, "Mstar": 3416000000.0, "hz_star": 0.55, "Rd_gas": 4.97, "Mgas": 2690590000.0000005, "hz_gas": 0.15},
{"name": "UGC06923", "Rd_star": 1.44, "Mstar": 1445000000.0, "hz_star": 0.29, "Rd_gas": 2.59, "Mgas": 1075970000.0, "hz_gas": 0.15},
{"name": "UGC06930", "Rd_star": 3.94, "Mstar": 4466000000.0, "hz_star": 0.79, "Rd_gas": 7.09, "Mgas": 4305210000.000001, "hz_gas": 0.15},
{"name": "UGC06973", "Rd_star": 1.07, "Mstar": 26935000000.0, "hz_star": 0.21, "Rd_gas": 1.93, "Mgas": 2331490000.0, "hz_gas": 0.15},
{"name": "UGC06983", "Rd_star": 3.21, "Mstar": 2649000000.0, "hz_star": 0.64, "Rd_gas": 5.78, "Mgas": 3946110000.0000005, "hz_gas": 0.15},
{"name": "UGC07089", "Rd_star": 2.26, "Mstar": 1792500000.0, "hz_star": 0.45, "Rd_gas": 4.07, "Mgas": 1614620000.0, "hz_gas": 0.15},
{"name": "UGC07125", "Rd_star": 3.38, "Mstar": 1356000000.0, "hz_star": 0.68, "Rd_gas": 6.08, "Mgas": 6156569999.999999, "hz_gas": 0.15},
{"name": "UGC07151", "Rd_star": 1.25, "Mstar": 1142000000.0, "hz_star": 0.25, "Rd_gas": 2.25, "Mgas": 819280000.0, "hz_gas": 0.15},
{"name": "UGC07232", "Rd_star": 0.29, "Mstar": 56500000.0, "hz_star": 0.06, "Rd_gas": 0.52, "Mgas": 61180000.00000001, "hz_gas": 0.15},
{"name": "UGC07261", "Rd_star": 1.2, "Mstar": 876500000.0, "hz_star": 0.24, "Rd_gas": 2.16, "Mgas": 1846040000.0, "hz_gas": 0.15},
{"name": "UGC07323", "Rd_star": 2.26, "Mstar": 2054500000.0, "hz_star": 0.45, "Rd_gas": 4.07, "Mgas": 960260000.0, "hz_gas": 0.15},
{"name": "UGC07399", "Rd_star": 1.64, "Mstar": 578000000.0, "hz_star": 0.33, "Rd_gas": 2.95, "Mgas": 990850000.0, "hz_gas": 0.15},
{"name": "UGC07524", "Rd_star": 3.46, "Mstar": 1218000000.0, "hz_star": 0.69, "Rd_gas": 6.23, "Mgas": 2366070000.0, "hz_gas": 0.15},
{"name": "UGC07559", "Rd_star": 0.58, "Mstar": 54500000.0, "hz_star": 0.12, "Rd_gas": 1.04, "Mgas": 224770000.00000003, "hz_gas": 0.15},
{"name": "UGC07577", "Rd_star": 0.9, "Mstar": 22500000.0, "hz_star": 0.18, "Rd_gas": 1.62, "Mgas": 58520000.0, "hz_gas": 0.15},
{"name": "UGC07603", "Rd_star": 0.53, "Mstar": 188000000.0, "hz_star": 0.11, "Rd_gas": 0.95, "Mgas": 343140000.0, "hz_gas": 0.15},
{"name": "UGC07608", "Rd_star": 1.5, "Mstar": 132000000.0, "hz_star": 0.3, "Rd_gas": 2.7, "Mgas": 711550000.0000001, "hz_gas": 0.15},
{"name": "UGC07690", "Rd_star": 0.57, "Mstar": 429000000.0, "hz_star": 0.11, "Rd_gas": 1.03, "Mgas": 518700000.00000006, "hz_gas": 0.15},
{"name": "UGC07866", "Rd_star": 0.61, "Mstar": 62000000.0, "hz_star": 0.12, "Rd_gas": 1.1, "Mgas": 156940000.0, "hz_gas": 0.15},
{"name": "UGC08286", "Rd_star": 1.05, "Mstar": 627500000.0, "hz_star": 0.21, "Rd_gas": 1.89, "Mgas": 853860000.0000001, "hz_gas": 0.15},
{"name": "UGC08490", "Rd_star": 0.67, "Mstar": 508499999.99999994, "hz_star": 0.13, "Rd_gas": 1.21, "Mgas": 957600000.0, "hz_gas": 0.15},
{"name": "UGC08550", "Rd_star": 0.45, "Mstar": 144500000.0, "hz_star": 0.09, "Rd_gas": 0.81, "Mgas": 383040000.0, "hz_gas": 0.15},
{"name": "UGC08699", "Rd_star": 3.09, "Mstar": 25151000000.0, "hz_star": 0.62, "Rd_gas": 5.56, "Mgas": 4971540000.0, "hz_gas": 0.15},
{"name": "UGC08837", "Rd_star": 1.72, "Mstar": 250500000.0, "hz_star": 0.34, "Rd_gas": 3.1, "Mgas": 425600000.00000006, "hz_gas": 0.15},
{"name": "UGC09037", "Rd_star": 4.28, "Mstar": 34307000000.000004, "hz_star": 0.86, "Rd_gas": 7.7, "Mgas": 25373740000.0, "hz_gas": 0.15},
{"name": "UGC09133", "Rd_star": 6.97, "Mstar": 141463000000.0, "hz_star": 1.39, "Rd_gas": 12.55, "Mgas": 44459240000.0, "hz_gas": 0.15},
{"name": "UGC09992", "Rd_star": 1.04, "Mstar": 168000000.0, "hz_star": 0.21, "Rd_gas": 1.87, "Mgas": 422940000.00000006, "hz_gas": 0.15},
{"name": "UGC10310", "Rd_star": 1.8, "Mstar": 870500000.0, "hz_star": 0.36, "Rd_gas": 3.24, "Mgas": 1590680000.0, "hz_gas": 0.15},
{"name": "UGC11455", "Rd_star": 5.93, "Mstar": 187161000000.0, "hz_star": 1.19, "Rd_gas": 10.67, "Mgas": 17735550000.000004, "hz_gas": 0.15},
{"name": "UGC11557", "Rd_star": 2.75, "Mstar": 6050500000.0, "hz_star": 0.55, "Rd_gas": 4.95, "Mgas": 3464650000.0, "hz_gas": 0.15},
{"name": "UGC11820", "Rd_star": 2.08, "Mstar": 485000000.0, "hz_star": 0.42, "Rd_gas": 3.74, "Mgas": 2629410000.0000005, "hz_gas": 0.15},
{"name": "UGC11914", "Rd_star": 2.44, "Mstar": 75014000000.0, "hz_star": 0.49, "Rd_gas": 4.39, "Mgas": 1181040000.0, "hz_gas": 0.15},
{"name": "UGC12506", "Rd_star": 7.38, "Mstar": 69785500000.0, "hz_star": 1.48, "Rd_gas": 13.28, "Mgas": 47289480000.0, "hz_gas": 0.15},
{"name": "UGC12632", "Rd_star": 2.42, "Mstar": 650500000.0, "hz_star": 0.48, "Rd_gas": 4.36, "Mgas": 2319520000.0000005, "hz_gas": 0.15},
{"name": "UGC12732", "Rd_star": 1.98, "Mstar": 833500000.0, "hz_star": 0.4, "Rd_gas": 3.56, "Mgas": 4867800000.000001, "hz_gas": 0.15},
{"name": "UGCA281", "Rd_star": 1.72, "Mstar": 97000000.0, "hz_star": 0.34, "Rd_gas": 3.1, "Mgas": 82460000.0, "hz_gas": 0.15},
{"name": "UGCA442", "Rd_star": 1.18, "Mstar": 70000000.0, "hz_star": 0.24, "Rd_gas": 2.12, "Mgas": 349790000.00000006, "hz_gas": 0.15},
{"name": "UGCA444", "Rd_star": 0.83, "Mstar": 6000000.0, "hz_star": 0.17, "Rd_gas": 1.49, "Mgas": 89110000.00000001, "hz_gas": 0.15},
]

# ---- UTILITIES ----
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name)

# ---- DATA DOWNLOAD ----
def download_and_extract_data():
    data_dir = "data/sparc"
    os.makedirs(data_dir, exist_ok=True)
    url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
    zip_path = os.path.join("data", "Rotmod_LTG.zip")
    existing_files = [f for f in os.listdir(data_dir) if f.endswith("_rotmod.dat")]
    if len(existing_files) < 5:
        print(">>> AUTOMATION: SPARC data missing. Initiating Download Sequence...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"!!! DOWNLOAD ERROR: {e}")

# ---- PHYSICS GRID & PROFILES ----
def safe_two_component_disk(n, Lbox, Rd_star, Mstar, hz_star, Rd_gas, Mgas, hz_gas):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x * x + y * y)
    dx = axis[1] - axis[0]

    def get_rho(M, Rd, hz):
        if M <= 0 or Rd <= 0:
            return np.zeros_like(R, dtype=np.float32)
        hz = max(hz, 1e-3)
        radial = np.exp(-R / Rd)
        z_scaled = np.abs(z / hz)
        vertical = np.zeros_like(z, dtype=np.float32)
        mask = z_scaled < 20.0
        vertical[mask] = (1.0 / np.cosh(z_scaled[mask]))**2
        rho0 = M / (4 * np.pi * Rd**2 * hz)
        return (rho0 * radial * vertical).astype(np.float32)

    rho_star = get_rho(Mstar, Rd_star, hz_star)
    rho_gas = get_rho(Mgas, Rd_gas, hz_gas)
    return rho_star + rho_gas, dx

def choose_box_and_grid(R_obs_max, L):
    target_half = max(1.5 * R_obs_max, 4.0 * L, 20.0)
    Lbox = float(target_half)

    debug_dx = os.environ.get("DEBUG_DX", None)
    if debug_dx is not None:
        try:
            dx_target = float(debug_dx)
            Lbox_debug = 80.0  # kpc - reduces memory and keeps physics-local
            Lbox = Lbox_debug

            n_req = int(round(2.0 * Lbox / dx_target))
            N_MIN = 64
            N_MAX = 512   # (safe for my personal laptop 32g RAM)
            n = max(N_MIN, min(N_MAX, n_req))
            if n % 2 == 1:
                n += 1

            if n != n_req:
                print(f"[WARN] requested dx={dx_target:.3g} => n_req={n_req} clipped -> n={n}")
            debug(f"[DEBUG] Forced dx={dx_target:.3g} => n={n}, Lbox={Lbox}")
            return Lbox, n
        except Exception:
            pass

    n = int(np.clip(round(2 * Lbox / 0.5), 64, 320))
    if n % 2 == 1:
        n += 1
    return Lbox, n


def build_U_grid(n, Lbox, L, kernel, beta=1.0):
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    # -------------------------------------------------------
    # Build analytic kernel (no normalization here!)
    # -------------------------------------------------------
    if kernel == "plummer":
        U = U_plummer(r, L)
    elif kernel == "exp-core":
        U = U_exp_core(r, L)
    elif kernel == "ananta-hybrid":
        U = U_ananta_hybrid(r, L, beta=beta)
    else:
        raise ValueError("kernel error")

    debug("[DBG-K] U.dtype, U.min, U.max, U.mean =",
          U.dtype, float(U.min()), float(U.max()), float(np.mean(U)))

    # apply smooth spherical taper/cut to avoid long-range grid leakage
    R = np.sqrt(x*x + y*y + z*z)
    R_cut = min(3.0 * float(L), 0.45 * float(Lbox))
    taper = np.ones_like(U, dtype=np.float64)

    r0 = 0.85 * R_cut
    mask = (R > r0) & (R <= R_cut)
    if np.any(mask):
        frac = (R[mask] - r0) / (R_cut - r0)
        taper[mask] = 0.5 * (1.0 + np.cos(np.pi * frac))

    taper[R > R_cut] = 0.0
    U = U * taper

    nonzero_frac = np.count_nonzero(taper) / taper.size
    fix(f"[TAPER] nonzero fraction = {nonzero_frac:.6f}")

    if nonzero_frac < 0.02:
        raise RuntimeError(
            f"[TAPER-FAIL] taper removed too much kernel: nonzero={nonzero_frac:.6f}"
        )

    # compute discrete integral and apply scale to enforce ∫U d^3r = 1/L exactly
    dx = float(axis[1] - axis[0])
    cell_vol = dx**3
    current_integral = float(np.sum(U) * cell_vol)
    desired_integral = 1.0 / max(1e-12, float(L))

    if not np.isfinite(current_integral) or abs(current_integral) < 1e-30:
        raise RuntimeError(
            f"Bad kernel integral {current_integral:.3e} for L={L} at dx={dx}"
        )

    scale = desired_integral / current_integral
    U *= scale

    fix(
        f"[FIX] kernel renormalized: integral {current_integral:.3e} "
        f"-> {float(np.sum(U) * cell_vol):.3e} (scale={scale:.6e})"
    )

    #DC guard & single-point zero
    U -= float(np.mean(U))
    U.flat[0] = 0.0


    return U.astype(np.float32)
U_CACHE = {}
def get_U_grid(n, Lbox, L, kernel, beta=1.0):
    key = (kernel, float(L), int(n), round(float(Lbox), 2), float(beta))
    if key not in U_CACHE:
        U_CACHE[key] = build_U_grid(n, Lbox, L, kernel, beta=beta)
    return U_CACHE[key]


def fill_nans(arr):
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    if np.all(mask):
        return arr
    idx = np.where(~mask)[0]
    arr[mask] = np.interp(np.where(mask)[0], idx, arr[idx])
    return arr

def radial_profile_2d(arr2d, dx, max_r, nbins=30):
    n = arr2d.shape[0]
    cx = cy = n // 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = np.sqrt(((xx - cx + 0.5) * dx)**2 + ((yy - cy + 0.5) * dx)**2).astype(np.float32)
    rb = np.linspace(0, max_r * 1.1, nbins + 1).astype(np.float32)
    centers = 0.5 * (rb[1:] + rb[:-1])
    prof = np.empty(nbins, dtype=np.float32)
    prof[:] = np.nan
    for i, (r0, r1) in enumerate(zip(rb[:-1], rb[1:])):
        m = (R >= r0) & (R < r1)
        if np.any(m):
            prof[i] = np.mean(arr2d[m])
    return centers, fill_nans(prof)

# ---- IO ----
def read_galaxy_table(path_csv):
    out = []
    if os.path.exists(path_csv):
        try:
            with open(path_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # --- FAST MODE FILTER: only keep the TARGET if it is set ---
                    if TARGET_GALAXY and row["name"].strip() != TARGET_GALAXY:
                        continue

                    def num(x):
                        try: return float(x)
                        except: return 0.0

                    g = {
                        "name":    row["name"].strip(),
                        "Rd_star": num(row.get("Rd_star_kpc", 0)),
                        "Mstar":   num(row.get("Mstar_Msun", 0)),
                        "hz_star": num(row.get("hz_star_kpc", "0.3")),
                        "Rd_gas":  num(row.get("Rd_gas_kpc", "0")),
                        "Mgas":    num(row.get("Mgas_Msun", "0")),
                        "hz_gas":  num(row.get("hz_gas_kpc", "0.15")),
                    }
                    # ---  helium correction for calculations ---
                    HELIUM_FACTOR = 1.33
                    g['Mgas_HI'] = float(g['Mgas'])            
                    g['Mgas'] = float(g['Mgas_HI']) * HELIUM_FACTOR
                    if g["Rd_gas"] <= 0:
                        g["Rd_gas"] = 1.8 * (g["Rd_star"] if g["Rd_star"] > 0 else 1.0)
                    out.append(g)
        except Exception as e:
            print(f"Note: Error reading CSV ({e}).")

    if not out:
        print(">>> Using hardcoded fallback (NGC3198)")
        return NIGHTMARE_FLEET

    return out


def try_read_observed_rc(name):
    base_dirs = ["data/sparc", "data/sparc/Rotmod_LTG"]
    file_to_read = None
    is_dat = False
    for d in base_dirs:
        path_dat = os.path.join(d, f"{name}_rotmod.dat")
        path_csv = os.path.join(d, f"{name}_rc.csv")
        if os.path.exists(path_dat):
            file_to_read, is_dat = path_dat, True
            break
        elif os.path.exists(path_csv):
            file_to_read, is_dat = path_csv, False
            break
    if file_to_read is None:
        return None

    R, V = [], []
    try:
        with open(file_to_read, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Rad"):
                    continue
                if is_dat:
                    parts = line.split()
                else:
                    parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        R.append(float(parts[0]))
                        V.append(float(parts[1]))
                    except ValueError:
                        continue
        if len(R) == 0:
            return None
        return np.array(R, dtype=np.float32), np.array(V, dtype=np.float32)
    except Exception:
        return None

# ---- CORE PIPELINE ----
def predict_rc_for_params(gal, L, mu, kernel, beta=1.0):
    """
    Compute predicted rotation curve for one galaxy.
    Added: beta (global amplitude correction) forwarded to kernel grid.
    """

    obs = try_read_observed_rc(gal["name"])
    if obs is None or obs[0].size == 0:
        return None

    R_obs_max = float(np.nanmax(obs[0]))
    Lbox, n = choose_box_and_grid(R_obs_max, L)

    rho, dx = safe_two_component_disk(
        n,
        Lbox,
        Rd_star=gal["Rd_star"],
        Mstar=gal["Mstar"],
        hz_star=gal["hz_star"],
        Rd_gas=gal["Rd_gas"],
        Mgas=gal["Mgas"],
        hz_gas=gal["hz_gas"]
    )
    # --- DEBUG / UNITS CHECK ---
    Sigma_kpc2 = np.sum(rho, axis=2) * dx    # Msun / kpc^2  (dx in kpc; rho in Msun/kpc^3)
    Sigma_pc2   = Sigma_kpc2 / 1e6           # Msun / pc^2
    debug(f"[DBG] GAL: {gal['name']}")
    debug("[DBG] Mstar (Msun):", gal['Mstar'], "Mgas_HI (Msun):", gal.get('Mgas_HI'), "Mgas_with_He:", gal['Mgas'])
    debug(f"[DBG] Rd_star,kpc: {gal['Rd_star']}, Rd_gas,kpc: {gal['Rd_gas']}, hz_star: {gal['hz_star']}, hz_gas: {gal['hz_gas']}")
    debug(f"[DBG] dx(kpc): {dx:.6e}, Sigma_kpc2 min/max: {Sigma_kpc2.min():.3e}/{Sigma_kpc2.max():.3e}, Sigma_pc2 min/max: {Sigma_pc2.min():.3e}/{Sigma_pc2.max():.3e}")

    G32 = np.float32(G)
    # ============================================================
    # SEPARATE STAR AND GAS POTENTIALS
    # ============================================================
    axis = np.linspace(-Lbox, Lbox, n, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(x * x + y * y)

    def get_rho_component(M, Rd, hz):
        if M <= 0 or Rd <= 0:
            return np.zeros_like(R, dtype=np.float32)
        hz = max(hz, 1e-3)
        radial = np.exp(-R / Rd)
        z_scaled = np.abs(z / hz)
        vertical = np.zeros_like(z, dtype=np.float32)
        maskz = z_scaled < 20.0
        vertical[maskz] = (1.0 / np.cosh(z_scaled[maskz]))**2
        rho0 = M / (4 * np.pi * Rd**2 * hz)
        return (rho0 * radial * vertical).astype(np.float32)

    rho_star = get_rho_component(gal["Mstar"], gal["Rd_star"], gal["hz_star"])
    rho_gas  = get_rho_component(gal["Mgas"],  gal["Rd_gas"],  gal["hz_gas"])

    # Newtonian potentials
    phi_star = phi_newtonian_from_rho(rho_star, Lbox, Gval=G32)
    gx_star, gy_star, _ = gradient_from_phi(phi_star, Lbox)

    phi_gas = phi_newtonian_from_rho(rho_gas, Lbox, Gval=G32)
    gx_gas, gy_gas, _ = gradient_from_phi(phi_gas, Lbox)

    # Total baryonic field 
    gx_b = gx_star + gx_gas
    gy_b = gy_star + gy_gas

    # --- Kernel: build and compute safely ---
    U = get_U_grid(n, Lbox, L, kernel, beta=beta)

    phi_K_raw = None
    phi_K = None
    gx_K = np.zeros_like(gx_b, dtype=np.float32)
    gy_K = np.zeros_like(gy_b, dtype=np.float32)

    try:
        # convolution -> raw kernel potential
        phi_K_raw = conv_fft(rho, U, zero_mode=True)
        # ---- EXTENDED FFT NORMALIZATION DIAGNOSTIC ----
        # compute dx/cell_vol to convert sums <-> integrals
        dx = float(axis[1] - axis[0])
        cell_vol = dx**3

        # basic sums for sanity
        sum_U = float(np.sum(U) * cell_vol)
        sum_rho = float(np.sum(rho) * cell_vol)
        sum_phiKraw = None
        try:
            sum_phiKraw = float(np.sum(phi_K_raw) * cell_vol)
        except Exception:
            pass

        # min/max checks
        U_min, U_max = float(np.min(U)), float(np.max(U))
        rho_min, rho_max = float(np.min(rho)), float(np.max(rho))

        debug(f"[FFTDBG] dx={dx:.3g} sum_U={sum_U:.6e} sum_rho={sum_rho:.6e} raw_phi_sum={sum_phiKraw}")
        debug(f"[FFTDBG] U_min/max={U_min:.6e}/{U_max:.6e} rho_min/max={rho_min:.6e}/{rho_max:.6e}")

        # quick functional tests: convolve a uniform field and a test delta-like field
        try:
            ones = np.ones_like(rho, dtype=rho.dtype)
            test_phi_ones = conv_fft(ones, U, zero_mode=True)
            test_sum_ones = float(np.sum(test_phi_ones) * cell_vol)
            debug(f"[FFTDBG] test_conv(ones) sum={test_sum_ones:.6e}")
        except Exception as e:
            debug(f"[FFTDBG] test_conv(ones) failed: {e}")

        try:
            # delta at center
            delta = np.zeros_like(rho, dtype=rho.dtype)
            cx = cy = n // 2
            delta[cx, cy, n // 2] = 1.0 / cell_vol   # unit integral
            test_phi_delta = conv_fft(delta, U, zero_mode=True)
            test_sum_delta = float(np.sum(test_phi_delta) * cell_vol)
            test_min, test_max = float(np.min(test_phi_delta)), float(np.max(test_phi_delta))
            debug(f"[FFTDBG] test_conv(delta) sum={test_sum_delta:.6e} min={test_min:.6e} max={test_max:.6e}")
        except Exception as e:
            debug(f"[FFTDBG] test_conv(delta) failed: {e}")

        # ---- FFT NORMALIZATION DIAGNOSTIC ----
        dx = float(axis[1] - axis[0])
        cell_vol = dx**3
        try:
            raw_sum = float(np.sum(phi_K_raw) * cell_vol)
            debug(f"[FFTDBG] dx={dx:.3g} raw_phi_sum={raw_sum:.6e}")
        except Exception:
            debug("[FFTDBG] conv_fft returned non-array or failed sum diagnostic")

        # polarity sign and coupling
        phi_K = (POLARITY_SIGN * mu * G32 * phi_K_raw).astype(np.float32)

        # gradients -> forces from kernel
        gx_K, gy_K, _ = gradient_from_phi(phi_K, Lbox)

    except Exception as e:
        print(f"[ERROR] Kernel computation failed for L={L}, mu={mu}: {e}")
    # ============================================================
    #  POLARITY DIAGNOSTIC + AUTO-FLIP
    # ============================================================
    iz = n // 2
    try:
        offset = int(round(10.0 / max(1e-6, (2.0 * Lbox) / float(n))))
        ix = n // 2 + offset
        iy = n // 2
        if 0 <= ix < n:
            _b = float(gx_b[ix, iy, iz])
            _k = float(gx_K[ix, iy, iz])
            if (_b > 0 and _k < 0) or (_b < 0 and _k > 0):
                print(f"[WARNING] Polarity mismatch at L={L}, mu={mu}: baryon={_b:.3e}, kernel={_k:.3e}")
    except Exception:
        pass

    # Global dot-test for polarity
    dot_prod = np.nansum(gx_b[:, :, iz] * gx_K[:, :, iz] +
                         gy_b[:, :, iz] * gy_K[:, :, iz])
    mag_bary = np.sqrt(np.nansum(gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2))
    mag_kern = np.sqrt(np.nansum(gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2))
    debug(f"[POLDBG] dot_prod={dot_prod:.6e}, mag_bary={mag_bary:.6e}, mag_kern={mag_kern:.6e}")

    if (dot_prod < 0) and (mag_kern > 0) and (abs(dot_prod) > 1e-6 * mag_bary * mag_kern):
        debug("[POLDBG] Anti-aligned fields detected. Flipping kernel sign.")
        gx_K[:, :, iz] = -gx_K[:, :, iz]
        gy_K[:, :, iz] = -gy_K[:, :, iz]

    # ============================================================
    #  RADIAL PROFILES — PER-COMPONENT g^2 MAPS
    # ============================================================
    g_total_sq = (gx_b[:, :, iz] + gx_K[:, :, iz])**2 + \
                 (gy_b[:, :, iz] + gy_K[:, :, iz])**2
    g_bary_sq = gx_b[:, :, iz]**2 + gy_b[:, :, iz]**2
    g_kern_sq = gx_K[:, :, iz]**2 + gy_K[:, :, iz]**2
    g_star_sq = gx_star[:, :, iz]**2 + gy_star[:, :, iz]**2
    g_gas_sq  = gx_gas[:, :, iz]**2  + gy_gas[:, :, iz]**2

    # ============================================================
    # AVERAGE g^2 THEN TAKE SQRT
    # ============================================================
    R_centers, g2_mean_total = radial_profile_2d(g_total_sq, dx, R_obs_max, nbins=RADIAL_BINS)
    _,         g2_mean_bary  = radial_profile_2d(g_bary_sq, dx, R_obs_max, nbins=RADIAL_BINS)
    _,         g2_mean_kern  = radial_profile_2d(g_kern_sq, dx, R_obs_max, nbins=RADIAL_BINS)
    _,         g2_mean_star  = radial_profile_2d(g_star_sq, dx, R_obs_max, nbins=RADIAL_BINS)
    _,         g2_mean_gas   = radial_profile_2d(g_gas_sq,  dx, R_obs_max, nbins=RADIAL_BINS)

    g_mean_total = np.sqrt(np.maximum(g2_mean_total, 0.0))
    g_mean_bary  = np.sqrt(np.maximum(g2_mean_bary,  0.0))
    g_mean_kern  = np.sqrt(np.maximum(g2_mean_kern,  0.0))
    g_mean_star  = np.sqrt(np.maximum(g2_mean_star,  0.0))
    g_mean_gas   = np.sqrt(np.maximum(g2_mean_gas,   0.0))

    # ============================================================
    #  VELOCITY FIELDS 
    # ============================================================
    v_total_model = np.sqrt(np.maximum(R_centers * g_mean_total, 0.0))
    v_baryons     = np.sqrt(np.maximum(R_centers * g_mean_bary,  0.0))
    v_kernel      = np.sqrt(np.maximum(R_centers * g_mean_kern,  0.0))
    v_star        = np.sqrt(np.maximum(R_centers * g_mean_star,  0.0))
    v_gas         = np.sqrt(np.maximum(R_centers * g_mean_gas,   0.0))

    # ============================================================
    #  AUTHORITATIVE QUADRATURE OVERWRITE 
    # ============================================================
    v_total = np.sqrt(np.maximum(v_star**2 + v_gas**2 + v_kernel**2, 0.0))
    v_combined = np.sqrt(np.maximum(v_baryons**2 + v_kernel**2, 0.0))
    maxdiff = np.nanmax(np.abs(v_total - v_combined))
    if maxdiff > 1e-6:
        fix(f"[FIX] Overwriting v_total by quadrature (diff={maxdiff:.3g})")
        v_total = v_combined

    # ============================================================
    #  BUILD OBSERVED VELOCITIES ON MODEL RADIAL GRID
    # ============================================================
    try:
        R_obs, V_obs = obs  
        v_obs = np.interp(R_centers, R_obs, V_obs, left=np.nan, right=np.nan)
    except Exception:
        print("[ERROR] Could not build v_obs; skipping alpha-fit")
        v_obs = np.full_like(R_centers, np.nan)
    # ============================================================
    #  ALPHA AMPLITUDE FIT 
    # ============================================================
    valid_obs = (~np.isnan(v_obs)) & (v_obs > 0)
    mask = valid_obs & (v_kernel > 1e-6)

    if np.count_nonzero(mask) >= 5:
        eps = 1e-8
        v_total_before = v_total.copy()

        num = (v_obs[mask]**2 - v_baryons[mask]**2)
        den = (v_kernel[mask]**2 + eps)
        alpha2 = np.nanmean(np.clip(num / den, 0, None))
        alpha = np.sqrt(alpha2) if alpha2 > 0 else 1.0
        debug(f"[ALPHA] alpha_raw={alpha:.4g}")

        # Apply alpha (auditable)
        v_kernel = v_kernel * alpha
        v_total  = np.sqrt(np.maximum(v_baryons**2 + v_kernel**2, 0.0))

        # error metrics for diagnostics
        err_before = np.nanmean((v_obs[mask] - v_total_before[mask])**2)
        err_after  = np.nanmean((v_obs[mask] - v_total[mask])**2)
        debug(f"[ALPHA] err_before={err_before:.4g}, err_after={err_after:.4g}")

    else:
        debug("[ALPHA] skipped — insufficient valid points for alpha-fit")

    # ---------------------------------------------------------------
    # POSTFIX DIAGNOSTICS / SANITY CHECKS
    # ---------------------------------------------------------------
    try:
        debug("[KDBG] U.mean(before)=", U_mean)
        debug("[KDBG] U.integral_before =", np.sum(U) * dx**3)
        debug("[KDBG] desired_integral =", 1.0/float(L))
        debug("[KDBG] U.scale_applied =", scale)
        debug("[KDBG] U.mean(after)=", float(np.mean(U)))
        debug("[KDBG] U.flat[0]=", float(U.flat[0]))
        debug("[DBG] POLARITY_SIGN =", POLARITY_SIGN)
        debug("[DIAG] U_mean_before =", U_mean)
        debug("[DIAG] U_integral_before =", current_integral)
        debug("[DIAG] U_integral_after =", float(np.sum(U) * dx**3))
        debug("[DIAG] U_scale_applied =", float(scale))
        debug("[DIAG] R_centers[:10] =", R_centers[:min(10, len(R_centers))])
        debug("[DIAG] g_mean_total[:6] =", g_mean_total[:6])
        debug("[DIAG] g_mean_bary[:6]  =", g_mean_bary[:6])
        debug("[DIAG] g_mean_kern[:6]  =", g_mean_kern[:6])
        debug("[DIAG] v_star[:5] =", v_star[:5] if 'v_star' in locals() else 'NA')
        debug("[DIAG] v_gas[:5]  =", v_gas[:5]  if 'v_gas'  in locals() else 'NA')
        debug("[DIAG] v_kernel[:5] =", v_kernel[:5])
        debug("[DIAG] polarity: dot_prod, mag_bary, mag_kern =", dot_prod, mag_bary, mag_kern)

    except Exception:
        pass

    # composition consistency check
    v_combined_post = np.sqrt(np.maximum(v_baryons**2 + v_kernel**2, 0.0))
    maxdiff2 = np.nanmax(np.abs(v_total - v_combined_post))
    if maxdiff2 > 1e-6:
        print(f"[WARN] v_total vs sqrt(v_baryons^2+v_kernel^2) mismatch: max diff = {maxdiff2:.6g} km/s")
    else:
        print("[OK] v_total composition check passed (quadrature).")

    # ---------------------------------------------------------------
    # Safe cleanup
    # ---------------------------------------------------------------
    for v in ("rho", "phi_b", "phi_K_raw", "phi_K",
              "gx_b", "gy_b", "gx_K", "gy_K", "g_total_sq"):
        if v in locals():
            try:
                del locals()[v]
            except Exception:
                pass

    gc.collect()
    return R_centers, v_total, v_baryons, v_kernel


def mafe(pred_at_R, obs_V):
    return float(np.median(np.abs(pred_at_R - obs_V) / np.clip(obs_V, 1e-6, None)))

# ---- MAIN ----
def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    download_and_extract_data()

    table_path = os.path.join("data", "galaxies.csv")
    gals = read_galaxy_table(table_path)

    log(f"Initializing H1 Surveyor for {len(gals)} galaxies...")
    mode = f"SINGLE({TARGET_GALAXY})" if TARGET_GALAXY is not None else "FULL"
    print(f"Mode: {mode}")

    all_best_params = {}
    summary = []

    for i, gal in enumerate(gals):
        safe_name = sanitize_filename(gal['name'])
        print(f"\n[{i+1}/{len(gals)}] Surveying {safe_name}...")
        obs = try_read_observed_rc(gal["name"])
        if obs is None:
            print(f"  -> MISSING DATA for {gal['name']}")
            continue

        R_obs, V_obs = obs
        local_best = {"L": None, "mu": None, "mafe": 1e99, "kernel": None}

        for kernel in KERNELS:
            for L in L_LIST:
                for mu in MU_LIST:
                    res = predict_rc_for_params(gal, L, mu, kernel, beta=1.15)
                    if res is None:
                        continue
                    R_pred, V_pred, _, _ = res
                    Vp = np.interp(R_obs, R_pred, V_pred, left=np.nan, right=np.nan)
                    m = np.isfinite(Vp)
                    if np.any(m):
                        score = mafe(Vp[m], V_obs[m])
                        if score < local_best["mafe"]:
                            local_best = {"L": float(L), "mu": float(mu), "mafe": score, "kernel": kernel}
                    gc.collect()

        if local_best["L"] is None:
            print(f"  -> Fit Failed for {gal['name']}")
            continue

        all_best_params[gal["name"]] = local_best
        summary.append({"name": gal["name"], "mafe": local_best["mafe"], "L": local_best["L"], "mu": local_best["mu"]})
        print(f"  -> Best Fit: L={local_best['L']} kpc, mu={local_best['mu']} (Error: {local_best['mafe']:.4f})")

        final_res = predict_rc_for_params(
            gal,
            local_best["L"],
            local_best["mu"],
            local_best["kernel"],
            beta=1.15
        )

        if final_res is None:
            continue
        R_pred, V_pred, V_b, V_k = final_res

        out = os.path.join("figs", f"rc_{safe_name}_best.png")
        plt.figure(figsize=(5, 4))
        plt.plot(R_pred, V_pred, "-", color='orange', lw=2, label='H1 Total')
        plt.plot(R_pred, V_b, ':', color='cyan', lw=1.5, label='Baryons')
        plt.plot(R_pred, V_k, '--', color='lime', lw=1.5, label=f'Kernel (L={local_best["L"]}, mu={local_best["mu"]})')
        if obs is not None:
            plt.plot(R_obs, V_obs, "o", color='white', mec='black', ms=4, mew=0.5, label="Observed")
        plt.xlabel("R [kpc]"); plt.ylabel("v [km/s]")
        plt.title(f"{gal['name']} — Best Fit")
        if obs is not None:
            plt.xlim(0, np.nanmax(R_obs) * 1.1)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        np.savetxt(
            os.path.join("results", f"rc_decomp_{safe_name}_best.csv"),
            np.c_[R_pred, V_b, V_k, V_pred],
            delimiter=",", header="R_kpc,V_baryon,V_kernel,V_total", comments=""
        )

        del R_pred, V_pred, V_b, V_k
        gc.collect()

    with open("results/all_galaxy_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2)
    with open("results/sparc_lite_summary.csv", "w") as f:
        f.write("name,mafe,best_L,best_mu\n")
        for row in summary:
            f.write(f"{row['name']},{row['mafe']},{row['L']},{row['mu']}\n")

if __name__ == "__main__":
    main()
