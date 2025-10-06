import csv

input_file = "data/RXNCONSO.RRF"
output_file = "data/rxnorm_mapping.csv"

# Include ALL relevant term types from all sources
allowed_ttys = {
    # RXNORM
    "IN", "PIN", "MIN", "BN", "SCD", "SBD", "SBDC", "SBDF", "SBDFP", "SBDG",
    "SCDC", "SCDF", "SCDFP", "SCDG", "SCDGP", "GPCK", "BPCK", "PSN", "SY",
    # MTHSPL
    "SU", "DP", "MTH_RXN_DP",
    # MTHCMSFRF
    "PT"
}

seen = set()

with open(input_file, "r", encoding="utf-8") as infile, \
        open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.reader(infile, delimiter="|")
    writer = csv.writer(outfile)
    writer.writerow(["rxcui", "type", "name", "name_norm"])

    for row in reader:
        if len(row) < 18:
            continue

        tty = row[12]
        name = row[14]
        suppress = row[16]

        if tty not in allowed_ttys:
            continue
        if suppress == "Y":
            continue

        rxcui = row[0]
        name_norm = name.lower().strip()

        key = (rxcui, tty, name_norm)
        if key in seen:
            continue
        seen.add(key)

        writer.writerow([rxcui, tty, name.strip(), name_norm])

print(f"✅ Saved cleaned mapping to {output_file}")