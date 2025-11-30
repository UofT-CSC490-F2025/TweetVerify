import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
input_file = folder / "Political_tweets.csv"
output_file = folder / "cleaned_output.csv"

with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile) 
    writer = csv.writer(outfile) 

    next(reader, None)

    for row in reader:
        if not row or all(cell.strip() == "" for cell in row):
            continue

        error = False
        userdesc=""
        text = ""
        if len(row) == 13:
            userdesc = row[2]
            text = row[9]
        elif len(row) > 13:
            userdesc = row[2]
            text = row[9]
        else:
            error = True

        if not error:
            if userdesc != "" and text != "":
                writer.writerow([userdesc, text])


print(f"✅ Cleaning completed! Output saved to: {output_file}")
