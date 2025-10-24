import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ========== 基础配置 ==========
API_KEY = os.getenv("OPEN_AI_API_KEY")

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
MODEL = "gpt-4o-mini"

INPUT_FILE = folder / "cleaned_output.csv"
OUTPUT_DIR = folder / "batch_jobs_parallel"
STATUS_LOG = folder / "batch_status.log"
FINAL_CSV = folder / "good_tweets_all.csv"

TOTAL_LIMIT = 500
MAX_PER_BATCH = 500
POLL_INTERVAL_SEC = 180
DOWNLOAD_MAX_RETRIES = 5
RETRY_WAIT_SEC = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = OpenAI(api_key=API_KEY)

QUALITY_PROMPT = (
    "You are a tweet quality classifier.\\n\\n"
    "Classify whether the following tweet is a high-quality, politically relevant, English-language tweet "
    "that expresses a clear political viewpoint, opinion, or reaction to current events.\\n\\n"
    "Respond ONLY with:\\n"
    "- \"GOOD\"  → if the tweet is politically relevant, meaningful, and suitable for political content generation.\\n"
    "- \"BAD\"   → if it is advertising, self-promotion, spam, irrelevant, extremely short, or lacks clear meaning.\\n\\n"
    "No punctuation, no explanation.\\n\\n"
    "Tweet follows below:\\n"
)


# ========== 工具函数 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def safe_sleep(seconds):
    for remaining in range(seconds, 0, -30):
        print(f"⏳ Waiting {remaining}s...", end="\r")
        time.sleep(min(30, remaining))
    print(" " * 50, end="\r")

def validate_jsonl(path):
    """确保每行是合法 JSON，无多余符号"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    raise ValueError(f"Empty line at {i}")
                json.loads(line)
        return True
    except Exception as e:
        log(f"❌ Invalid JSONL in {path}: {e}")
        return False

def append_good_rows_to_csv(rows):
    if not rows:
        return
    mode = "a" if os.path.exists(FINAL_CSV) else "w"
    with open(FINAL_CSV, mode, encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(["custom_id", "label"])
        writer.writerows(rows)

# ========== 1) 生成分片 JSONL ==========
log("📦 Generating split JSONL files...")

batch_idx, total_processed = 0, 0
part_data = []

with open(INPUT_FILE, encoding="utf-8", errors="ignore") as fin:
    reader = csv.reader(fin)
    next(reader, None)

    for row in reader:
        if total_processed >= TOTAL_LIMIT:
            break
        if len(row) < 2:
            continue

        tweet_text = row[1].strip()
        if not tweet_text:
            continue

        part_data.append(tweet_text)
        total_processed += 1

        if len(part_data) >= MAX_PER_BATCH:
            batch_idx += 1
            out_path = os.path.join(OUTPUT_DIR, f"batch_input_{batch_idx}.jsonl")
            with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
                for i, txt in enumerate(part_data, 1):
                    obj = {
                        "custom_id": f"batch{batch_idx}_tweet_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "temperature": 0,
                            "messages": [
                                {"role": "system", "content": QUALITY_PROMPT},
                                {"role": "user", "content": txt}
                            ]
                        }
                    }
                    line = json.dumps(obj, ensure_ascii=False)
                    # ✅ 再次验证每条 JSON
                    json.loads(line)
                    fout.write(line + "\n")

            if validate_jsonl(out_path):
                log(f"✅ JSON validation passed for {out_path}")
            else:
                log(f"❌ JSON validation failed for {out_path}")
                raise SystemExit(1)

            log(f"✅ Created {len(part_data)} entries → {out_path}")
            part_data.clear()

# 写最后一批
if part_data:
    batch_idx += 1
    out_path = os.path.join(OUTPUT_DIR, f"batch_input_{batch_idx}.jsonl")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        for i, txt in enumerate(part_data, 1):
            obj = {
                "custom_id": f"batch{batch_idx}_tweet_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": QUALITY_PROMPT},
                        {"role": "user", "content": txt}
                    ]
                }
            }
            line = json.dumps(obj, ensure_ascii=False)
            json.loads(line)
            fout.write(line + "\n")

    if validate_jsonl(out_path):
        log(f"✅ JSON validation passed for {out_path}")
    else:
        log(f"❌ JSON validation failed for {out_path}")
        raise SystemExit(1)
    log(f"✅ Created {len(part_data)} entries → {out_path}")

log(f"📊 Total batches created: {batch_idx}")

# ========== 2) 上传并一次性创建所有 batch ==========
batch_records = {}  # idx -> {"id": str|None, "status": str, "input_file": str|None}

for i in range(1, batch_idx + 1):
    path = os.path.join(OUTPUT_DIR, f"batch_input_{i}.jsonl")

    # 终极防线：上传前再次校验
    if not validate_jsonl(path):
        log(f"❌ [Batch {i}] JSON invalid; skipping upload.")
        batch_records[i] = {"id": None, "status": "failed", "input_file": None}
        continue

    try:
        log(f"📤 [Batch {i}] Uploading {path}...")
        with open(path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="batch")
        log(f"✅ [Batch {i}] Uploaded file ID: {uploaded_file.id}")

        log(f"🚀 [Batch {i}] Creating batch job (24h window)...")
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_records[i] = {"id": batch.id, "status": "validating", "input_file": uploaded_file.id}
        log(f"✅ [Batch {i}] Created ID: {batch.id}")
    except Exception as e:
        log(f"❌ [Batch {i}] Creation failed: {e}")
        log(traceback.format_exc())
        batch_records[i] = {"id": None, "status": "failed", "input_file": None}

# ========== 3) 并行监控所有 batch，完成即下载解析 ==========
log("🛰️ All batch jobs submitted. Entering monitoring mode...")

completed_batches, failed_batches = set(), set()

def download_with_retries(file_id: str, save_path: str) -> bool:
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            log(f"📥 Downloading result (try {attempt}/{DOWNLOAD_MAX_RETRIES})...")
            result = client.files.content(file_id)
            with open(save_path, "wb") as f:
                f.write(result.read())
            log(f"✅ Saved result → {save_path}")
            return True
        except Exception as e:
            log(f"⚠️ Download failed: {e}")
            if attempt < DOWNLOAD_MAX_RETRIES:
                safe_sleep(RETRY_WAIT_SEC)
    return False

while True:
    all_done = True
    for idx, record in batch_records.items():
        if record["status"] in ("completed", "failed", "cancelled", "expired"):
            continue  # 已终态，不再查询
        if not record["id"]:
            failed_batches.add(idx)
            continue

        all_done = False
        try:
            current = client.batches.retrieve(record["id"])
            record["status"] = current.status
            log(f"📡 [Batch {idx}] Status: {current.status} | Requests: {current.request_counts}")

            if current.status == "completed":
                # 立即下载与解析
                result_file = os.path.join(OUTPUT_DIR, f"batch_result_{idx}.jsonl")
                download_url = f"https://api.openai.com/v1/files/{current.output_file_id}/content"
                log(f"🔗 [Batch {idx}] Official download URL:\n{download_url}")

                if download_with_retries(current.output_file_id, result_file):
                    # 解析 GOOD 结果并即时写入
                    good_rows = []
                    with open(result_file, "r", encoding="utf-8") as fin:
                        for line in fin:
                            try:
                                data = json.loads(line)
                                content = data["response"]["body"]["choices"][0]["message"]["content"].strip().upper()
                                if content == "GOOD":
                                    good_rows.append([data["custom_id"], content])
                            except Exception:
                                continue

                    append_good_rows_to_csv(good_rows)
                    log(f"💾 [Batch {idx}] Appended {len(good_rows)} GOOD tweets to {FINAL_CSV}")

                completed_batches.add(idx)

            elif current.status in ("failed", "cancelled", "expired"):
                failed_batches.add(idx)

        except Exception as e:
            log(f"⚠️ [Batch {idx}] Poll error: {e}")

    if all_done:
        break

    safe_sleep(POLL_INTERVAL_SEC)

# ========== 4) 最终报告 ==========
log("📊 All batches processed.")
if failed_batches:
    log(f"❌ Failed batches: {sorted(list(failed_batches))}")
else:
    log("✅ All batches completed successfully.")
log(f"📁 Final GOOD tweets file: {FINAL_CSV}")
