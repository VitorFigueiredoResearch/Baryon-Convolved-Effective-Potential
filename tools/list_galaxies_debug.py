from run_sparc_lite import read_galaxy_table
import os

csv_path = os.path.join("data", "galaxies.csv")
if not os.path.exists(csv_path):
    print("No data/galaxies.csv found at", csv_path)
else:
    table = read_galaxy_table(csv_path)
    print("Found", len(table), "entries. Here are the names (raw):")
    for g in table:
        print("-", repr(g.get("name","<no name>")))
