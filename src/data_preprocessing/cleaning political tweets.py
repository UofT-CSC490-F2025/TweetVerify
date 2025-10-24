import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
input_file = folder / "Political_tweets.csv"
output_file = folder / "cleaned_output.csv"

with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)  # 输入是以 \t 分隔
    writer = csv.writer(outfile)  # 输出同样以 \t 分隔

    next(reader, None)

    for row in reader:
        # 跳过空行
        if not row or all(cell.strip() == "" for cell in row):
            continue

        error = False
        userdesc=""
        text = ""
        # 按列数判断要取哪几列
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


print(f"✅ 清洗完成！输出已保存到：{output_file}")
