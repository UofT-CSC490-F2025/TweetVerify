import csv

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"

input_file = folder / "good_tweets_async.csv"

output_file = ROOT / "datalake" / "curated" / "twitter" / "high_quality_human.csv"

# ======== 主逻辑 ========
with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 读取表头
    header = next(reader)

    # 找出variant_index所在列的位置
    try:
        idx_variant = header.index("tweet")

    except ValueError:
        raise ValueError("❌ CSV中没有找到 'tweet' 列，请检查列名。")

    writer.writerow(["text", "label"])

    # 挨行读取并筛选
    for row in reader:
        if not row:
            continue

        text = row[idx_variant]
        label = 0
        writer.writerow([text, label])

print(f"📁 输出文件：{output_file}")