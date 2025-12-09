#!/usr/bin/env python3
# extract_sparc_params_v5.py
# Final SPARC -> H1 extractor 
#
# Requirements:
# - Place this script in the same folder as:
#   Table1.mrt, Bulges.mrt (if available), table2.mrt (optional), wise_ii table1.mrt (optional), rotmod/ (optional)
# - Run with Python 3 (Anaconda environment recommended).
#
# Outputs:
# - galaxies.csv               (columns: name,Rd_star,Mstar,hz_star,Rd_gas,Mgas,hz_gas)
# - galaxies_h1_lines.json     (one JSON object per line, exact H1 row format)
# - missing_report.txt         (human-readable list of galaxies with 0.0 fields)
#
# Author: adapted for Vítor Figueiredo

import json
import re
from pathlib import Path
import zipfile

# -------------------------
# Paths & filenames
# -------------------------
TABLE1 = Path("table1.mrt")
BULGES = Path("Bulges.mrt")  
TABLE2 = None          
WISE2  = Path("wise_ii table1.mrt")    # optional fallback
ROTMOD = Path("rotmod")                # folder with .dat files

OUT_CSV = Path("galaxies.csv")
OUT_JSON_LINES = Path("galaxies_h1_lines.json")
MISSING_REPORT = Path("missing_report.txt")

# -------------------------
# SPARC / H1 constants
# -------------------------
ML_DISK = 0.5
ML_BULGE = 0.7
HELIUM = 1.33
HZ_GAS = 0.15
HZ_STAR_FACTOR = 0.2
RD_GAS_FACTOR = 1.8

EXPECTED_N_GALAXIES = 175

# -------------------------
# Byte slices (0-based Python)
# from your header:
#  Name    : bytes  1-11   -> [0:11]
#  L[3.6]  : bytes 35-41   -> [34:41]
#  Rdisk   : bytes 62-66   -> [61:66]
#  MHI     : bytes 75-81   -> [74:81]
# -------------------------
SLICE_NAME = (0, 11)
SLICE_L36  = (34, 41)
SLICE_RD   = (61, 66)
SLICE_MHI  = (74, 81)

# -------------------------
# Helper functions
# -------------------------
def try_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s in ("-", "—"):
        return None
    s2 = s.replace(",", "")
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s2)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None

def canonical(n):
    if n is None:
        return ""
    return re.sub(r'\s+', '', str(n)).upper()

def read_text_file(path):
    if not path.exists():
        return []
    return path.read_text(encoding='utf-8', errors='ignore').splitlines()
def parse_table1_table(path):
    lines = read_text_file(path)
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln.startswith('#') or ln.lower().startswith("title"):
            continue

        parts = ln.split()
        if len(parts) < 15:
            continue

        name = parts[0]
        L36  = try_float(parts[7])
        Rd   = try_float(parts[11])
        MHI  = try_float(parts[13])

        rows.append({
            "name": name,
            "key": canonical(name),
            "L36": L36,
            "Rd": Rd,
            "MHI": MHI,
            "raw": ln
        })

    return rows
def parse_bulges(path):
    if not path.exists():
        return {}
    lines = read_text_file(path)
    mapping = {}
    for ln in lines:
        if not ln.strip():
            continue
        if ln.strip().lower().startswith("title") or "byte-by-byte" in ln.lower():
            continue
        name = ln[0:11].strip()
        if not name:
            parts = ln.strip().split()
            if len(parts) >= 2:
                name = parts[0]
            else:
                continue
        key = canonical(name)
        cand = [ln[11:19], ln[34:41], ln[18:26], ln[12:20]]
        Lb = None
        for c in cand:
            Lb = try_float(c)
            if Lb is not None:
                break
        if Lb is None:
            parts = ln.strip().split()
            nums = [try_float(p) for p in parts[1:] if try_float(p) is not None]
            if nums:
                Lb = nums[0]
        if Lb is not None:
            mapping[key] = Lb
    return mapping
def parse_wise(path):
    if not path.exists():
        return {}
    lines = read_text_file(path)
    mapping = {}
    for ln in lines:
        if not ln.strip():
            continue
        if ln.strip().lower().startswith("title") or "byte-by-byte" in ln.lower():
            continue
        parts = ln.strip().split()
        if len(parts) >= 3:
            name = parts[0]
            logM = try_float(parts[2])
            if logM is not None:
                mapping[canonical(name)] = 10 ** logM
                continue
        name_fw = ln[0:11].strip()
        logM_fw = try_float(ln[18:26])
        if name_fw and logM_fw is not None:
            mapping[canonical(name_fw)] = 10 ** logM_fw
    return mapping
def scan_rotmod_rd(folder):
    rdmap = {}
    if not folder.exists():
        return rdmap
    for f in folder.glob("*"):
        if not f.is_file():
            continue
        key = canonical(f.stem)
        try:
            text = f.read_text(errors="ignore")
        except:
            continue
        found = None
        for ln in text.splitlines():
            low = ln.lower()
            if "rd" in low or "rdisk" in low or "scale length" in low:
                v = try_float(ln)
                if v is not None:
                    found = v
                    break
        if found is not None:
            rdmap[key] = found
    # check zipped rotmod if present
    for z in Path(".").glob("rotmod*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as zf:
                for zi in zf.namelist():
                    stem = Path(zi).stem
                    key = canonical(stem)
                    if key in rdmap:
                        continue
                    try:
                        txt = zf.read(zi).decode("utf-8", errors="ignore")
                    except:
                        continue
                    for ln in txt.splitlines():
                        if "rd" in ln.lower() or "rdisk" in ln.lower():
                            v = try_float(ln)
                            if v is not None:
                                rdmap[key] = v
                                break
        except Exception:
            pass
    return rdmap
def compute_h1(name, L36, Lbulge, Rd, MHI):
    # apply SPARC rules
    Mstar = 0.0
    if L36 is not None:
        Mstar += ML_DISK * L36 * 1e9
    # bulge (if present)
    if Lbulge is not None:
        Mstar += ML_BULGE * Lbulge * 1e9
    # fallback not to invent: Mstar remains 0.0 if none available
    Mgas = 0.0
    if MHI is not None:
        Mgas = HELIUM * MHI * 1e9
    # scale lengths
    if Rd is not None:
        Rd_star = round(Rd, 2)
        hz_star = round(HZ_STAR_FACTOR * Rd, 2)
        Rd_gas = round(RD_GAS_FACTOR * Rd, 2)
    else:
        Rd_star = 0.0
        hz_star = 0.0
        Rd_gas = 0.0
    return {
        "name": name,
        "Rd_star": Rd_star,
        "Mstar": float(Mstar),
        "hz_star": hz_star,
        "Rd_gas": Rd_gas,
        "Mgas": float(Mgas),
        "hz_gas": HZ_GAS
    }

# -------------------------
# Run
# -------------------------
def main():
    if not TABLE1.exists():
        print("ERROR: Table1.mrt not found in current folder.")
        return

    t1_rows = parse_table1_table(TABLE1)
    print(f"Parsed {len(t1_rows)} candidate lines from Table1.mrt (raw parsing).")

    bmap = parse_bulges(BULGES) if BULGES.exists() else {}
    t2map = {}
    wmap = parse_wise(WISE2) if WISE2.exists() else {}
    rdmap = scan_rotmod_rd(ROTMOD)
    entries = []
    missing_report = []

    # Many SPARC files may include header lines; ensure we only include real galaxies by checking that at least one of L36,Rd,MHI is numeric or name looks like a typical galaxy code
    # Valid-name heuristic: contains letters and digits and not everything numeric
    def looks_like_galaxy(s):
        return bool(re.search(r'[A-Za-z]', s)) and len(s) >= 2

    for rec in t1_rows:
        name = rec["name"]
        if not looks_like_galaxy(name):
            continue
        key = rec["key"]
        L36 = rec["L36"]
        Rd  = rec["Rd"]
        MHI = rec["MHI"]

        # fallback Rd from rotmod
        if (Rd is None or Rd == 0) and key in rdmap:
            Rd = rdmap[key]

        # fallback stellar (WISE) -> implied Ldisk if L36 missing
        Lbulge = bmap.get(key)
        if L36 is None and key in wmap:
            implied_Mstar = wmap[key]
            # assume this mass is disk-like for fallback (we don't know bulge split)
            L36 = implied_Mstar / (ML_DISK * 1e9)

        # compute
        h1 = compute_h1(name, L36, Lbulge, Rd, MHI)
        entries.append(h1)

        missing = []
        if h1["Mstar"] == 0.0:
            missing.append("Mstar")
        if h1["Mgas"] == 0.0:
            missing.append("Mgas")
        if h1["Rd_star"] == 0.0:
            missing.append("Rd_star")
        if missing:
            missing_report.append((name, missing))

    # sort by name (stable)
    entries.sort(key=lambda x: x["name"].upper())

    # write CSV
    import csv
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name","Rd_star","Mstar","hz_star","Rd_gas","Mgas","hz_gas"])
        for r in entries:
            # Mstar and Mgas as floats; format Mstar with scientific notation if large
            w.writerow([
                r["name"],
                f"{r['Rd_star']:.2f}",
                format_sci(r["Mstar"]),
                f"{r['hz_star']:.2f}",
                f"{r['Rd_gas']:.2f}",
                format_sci(r["Mgas"]),
                f"{r['hz_gas']:.2f}"
            ])

    # write one-line-per-json
    with OUT_JSON_LINES.open("w", encoding="utf-8") as fh:
        for r in entries:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # missing report
    with MISSING_REPORT.open("w", encoding="utf-8") as fh:
        fh.write("Galaxies with fields set to 0.0 (could not be recovered):\n\n")
        for nm, flags in missing_report:
            fh.write(f"{nm}: {', '.join(flags)}\n")

    print("Done.")
    print("Processed entries (final):", len(entries))
    print("Missing entries:", len(missing_report))
    if len(entries) != EXPECTED_N_GALAXIES:
        print("WARNING: final galaxy count differs from expected", EXPECTED_N_GALAXIES)
        print("If the count is > expected, check for extra lines in Table1.mrt; if < expected, ensure Table1 is the official SPARC file.")
    else:
        print("Galaxy count matches expected 175.")

def format_sci(x):
    try:
        xf = float(x)
    except:
        return "0.0"
    # use compact scientific notation for big numbers
    if abs(xf) >= 1e9:
        return f"{xf:.6g}"
    else:
        return f"{xf:.6f}"

if __name__ == "__main__":
    main()
