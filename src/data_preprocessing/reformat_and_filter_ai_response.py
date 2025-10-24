import csv

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
input_file = folder / "ai_generated_realtime.csv"
output_file = ROOT / "datalake" / "curated" / "llm" / "ai_generated.csv"

# ======== 主逻辑 ========
with open(input_file, "r", encoding="utf-8", newline="") as infile, \
        open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 读取表头
    header = next(reader)

    # 找出variant_index所在列的位置
    try:
        idx_variant = header.index("variant_index")

    except ValueError:
        raise ValueError("❌ CSV中没有找到 'variant_index' 列，请检查列名。")

    writer.writerow(["text", "label"])

    # 挨行读取并筛选
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

print(f"✅ 完成筛选：总计 {count_in} 条，保留 {count_out} 条 (variant_index ∈ [1,5])。")
print(f"📁 输出文件：{output_file}")