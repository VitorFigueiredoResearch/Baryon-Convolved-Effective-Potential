#!/usr/bin/env python3
"""
diagnose_table1.py

One-click diagnostic tool to detect:
- which .mrt file contains the REAL SPARC galaxy data
- which .mrt files are metadata-only
- whether L[3.6], Rd, MHI slices contain real numbers
- why Mstar = 0 in your extraction

Outputs:
- table1_diagnostic.log   (complete human-readable report)
"""

from pathlib import Path
import re

# Fixed byte slices from SPARC Table1 spec
SL_NAME = slice(0, 11)     # bytes 1–11 (galaxy name)
SL_L36  = slice(34, 41)    # bytes 35–41 (L[3.6])
SL_RD   = slice(61, 66)    # bytes 62–66 (Rd)
SL_MHI  = slice(74, 81)    # bytes 75–81 (M_HI)

LOGFILE = Path("table1_diagnostic.log")

def try_float(s):
    if not s: return None
    s = s.strip()
    if s == "": return None
    # simple numeric regex
    m = re.match(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", s, flags=re.IGNORECASE)
    if not m: return None
    try:
        return float(m.group(0))
    except:
        return None

def looks_like_data_line(ln):
    # a real galaxy row typically starts with: letters+digits+spaces
    return re.match(r"^[A-Za-z0-9]{2,}", ln) is not None and not ln.strip().startswith("Note")

def analyze_file(path):
    lines = path.read_text(errors="ignore").splitlines()
    data_candidates = []
    extracted = []

    for ln in lines:
        if not looks_like_data_line(ln):
            continue

        name = ln[SL_NAME].strip()
        if not name:
            continue

        L36 = try_float(ln[SL_L36])
        Rd  = try_float(ln[SL_RD])
        MHI = try_float(ln[SL_MHI])

        data_candidates.append(name)
        extracted.append((name, L36, Rd, MHI, ln))

    real_rows = [row for row in extracted if any(v is not None for v in row[1:4])]

    return {
        "file": path.name,
        "total_lines": len(lines),
        "candidate_rows": len(data_candidates),
        "real_data_rows": len(real_rows),
        "samples": real_rows[:12],   # show first 12 rows
        "all_rows": extracted,
    }

def main():
    mrt_files = list(Path(".").glob("*.mrt"))
    if not mrt_files:
        print("No .mrt files found in this folder.")
        return

    report = []

    report.append("=== SPARC TABLE1 DIAGNOSTIC REPORT ===\n")

    for f in mrt_files:
        report.append(f"\n--- FILE: {f.name} ---")
        try:
            info = analyze_file(f)
        except Exception as e:
            report.append(f"Error reading file: {e}")
            continue

        report.append(f"Total lines: {info['total_lines']}")
        report.append(f"Candidate galaxy-like rows: {info['candidate_rows']}")
        report.append(f"Rows with real numeric data (L36, Rd, or MHI): {info['real_data_rows']}")

        if info["real_data_rows"] == 0:
            report.append("❌ This file DOES NOT contain SPARC galaxy data.")
        elif info["real_data_rows"] < 50:
            report.append("⚠️ This file contains SOME numeric rows but is NOT the full SPARC dataset (should be ~175 rows).")
        else:
            report.append("✅ This file appears to be the REAL SPARC Table1 data.")

        # show sample extracts
        report.append("\nSample rows (name | L36 | Rd | MHI):\n")
        for (name, L36, Rd, MHI, ln) in info["samples"]:
            report.append(f"{name:10} | L36={L36} | Rd={Rd} | MHI={MHI}")

    # Save log
    LOGFILE.write_text("\n".join(report), encoding="utf-8")
    print("Diagnostic complete. See table1_diagnostic.log for full report.")

if __name__ == "__main__":
    main()
