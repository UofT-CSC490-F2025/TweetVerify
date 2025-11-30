import csv

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
input_file = folder / "ai_generated_realtime.csv"
output_file = ROOT / "datalake" / "curated" / "llm" / "ai_generated.csv"

# ======== Main Logic ========
with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read header row
    header = next(reader)

    # Find the column index for 'variant_index'
    try:
        idx_variant = header.index("variant_index")

    except ValueError:
        raise ValueError("❌ Column 'variant_index' not found in CSV. Please check column names.")

    writer.writerow(["text", "label"])

    # Process each row and filter by variant_index
    count_in, count_out = 0, 0
    for row in reader:
        if not row or len(row) <= idx_variant:
            continue
        try:
            v = int(row[idx_variant])
        except ValueError:
            continue
        count_in += 1
        if 1 <= v <= 5:
            text = row[idx_variant+1]
            label = 1
            writer.writerow([text, label])
            count_out += 1

print(f"✅ Filtering completed: Total {count_in} rows, kept {count_out} rows (variant_index ∈ [1,5]).")
print(f"📁 Output file: {output_file}")