import csv

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"

input_file = folder / "good_tweets_async.csv"

output_file = ROOT / "datalake" / "curated" / "twitter" / "high_quality_human.csv"

# ======== Main Logic ========
with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read header row
    header = next(reader)

    # Find the column index for 'tweet'
    try:
        idx_variant = header.index("tweet")

    except ValueError:
        raise ValueError("❌ Column 'tweet' not found in CSV. Please check column names.")

    writer.writerow(["text", "label"])

    # Process each row and extract tweet text
    for row in reader:
        if not row:
            continue

        text = row[idx_variant]
        label = 0
        writer.writerow([text, label])

print(f"📁 Output file: {output_file}")