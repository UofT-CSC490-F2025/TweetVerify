import asyncio
import csv
import heapq
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

# ========== Basic Configuration ==========
API_KEY = os.getenv("OPEN_AI_API_KEY")

ROOT = Path(__file__).resolve().parent.parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"
INPUT_FILE = folder / "good_tweets_async_copy.csv"
OUTPUT_FILE = folder / "ai_generated_realtime.csv"

PROGRESS_FILE = folder / "progress_variants_paraphrase.txt"
STATUS_LOG = folder / "variants_status_paraphrase.log"
SAVED_ROWS_FILE = folder / "saved_rows_paraphrase.json"
SAVED_BATCHES_FILE = folder / "saved_batches_paraphrase.json"

MODEL = "gpt-5-mini"  # Recommended to use an authorized model first

# ========== Runtime Parameters ==========
LIMIT = 5000
CONCURRENCY = 50
RATE_LIMIT = 480
WINDOW = 60
MAX_RETRIES = 5
RETRY_DELAY = 2
BATCH_SIZE = 20
FORCE_SAVE_INTERVAL = 60
EXPECTED_VARIANT_COUNT = 5
starting_point = 0

client = AsyncOpenAI(api_key=API_KEY)
writer_queue = asyncio.Queue()   # ✅ Global queue definition

PROMPT_TEMPLATE_MULTI = """You are a professional political content generator.

For each tweet below, generate 5 variants that:
- Keep the same political stance and viewpoint.
- Keep the same emotional intensity and tone.
- Preserve style, punctuation, capitalization, and emoji usage.

Return ONLY valid JSON with this shape:
{
  "results": [
    {"row": <row_index>, "variants": ["v1","v2","v3","v4","v5"]},
    {"row": <row_index>, "variants": ["v1","v2","v3","v4","v5"]}
  ]
}
No commentary.

Tweets:
"""

# ========== Utility Functions ==========
def log(msg: str):
    STATUS_LOG.touch(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_progress(n: int):
    PROGRESS_FILE.touch(exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(n))

def read_progress() -> int:
    if os.path.exists(PROGRESS_FILE):
        try:
            return int(open(PROGRESS_FILE, "r").read().strip())
        except:
            return 0
    return 0

def load_json_set(path: str) -> set:
    if os.path.exists(path):
        try:
            return set(json.load(open(path, "r", encoding="utf-8")))
        except:
            return set()
    return set()

def save_json_set(path: str, s: set):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(s)), f)

def append_rows(rows):
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(["user_description", "original_tweet", "variant_index", "variant_text"])
        writer.writerows(rows)

# ========== Rate Limiting Control ==========
timestamps = deque()
async def rate_limit_guard():
    now = time.time()
    timestamps.append(now)
    while timestamps and timestamps[0] < now - WINDOW:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT:
        sleep_time = WINDOW - (now - timestamps[0]) + 0.1
        log(f"⏳ Throttling {sleep_time:.2f}s to stay under {RATE_LIMIT} RPM")
        await asyncio.sleep(sleep_time)

# ========== Batch Task Structure ==========
class BatchTask:
    def __init__(self, batch_id: int, items: list[tuple[int, str, str]], attempt: int = 1):
        self.priority = batch_id
        self.batch_id = batch_id
        self.items = items
        self.attempt = attempt
    def __lt__(self, other):
        return self.priority < other.priority

# ========== API Call Functions ==========
def _build_multi_prompt(items):
    lines = []
    for row_idx, _ud, tweet in items:
        safe_tweet = tweet.replace("\n", " ")
        lines.append(f"{row_idx}. {safe_tweet}")
    return PROMPT_TEMPLATE_MULTI + "\n".join(lines)

async def generate_variants_batch(items):
    await rate_limit_guard()
    prompt = _build_multi_prompt(items)
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}]
    )
    raw = completion.choices[0].message.content.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("No JSON object found in response")
    data = json.loads(m.group(0))
    res = data.get("results", [])
    out = {}
    if isinstance(res, list):
        for obj in res:
            if not isinstance(obj, dict):
                continue
            row_idx = obj.get("row")
            variants = obj.get("variants", [])
            if isinstance(row_idx, int) and isinstance(variants, list) and len(variants) >= EXPECTED_VARIANT_COUNT:
                out[row_idx] = [v.strip().replace("\n", " ").replace('"', "'") for v in variants[:EXPECTED_VARIANT_COUNT]]
    missing = [r for r, _, _ in items if r not in out]
    if missing:
        raise ValueError(f"Missing rows: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return out

async def writer_task(progress):
    log("🟢 Writer started")
    next_expected = read_progress() + 1
    saved_rows = load_json_set(SAVED_ROWS_FILE)
    last_save_time = time.time()

    while True:
        item = await writer_queue.get()
        if item == "STOP":
            break
        _, batch_id, batch_items_rows = item
        received = {}
        for row_idx, csv_rows in batch_items_rows:
            if row_idx not in saved_rows:
                received[row_idx] = csv_rows

        wrote_any = False
        merged_rows = []
        while next_expected in received:
            merged_rows.extend(received.pop(next_expected))
            saved_rows.add(next_expected)
            next_expected += 1
            wrote_any = True

        if wrote_any:
            append_rows(merged_rows)
            progress_row = (next_expected - 1) * BATCH_SIZE + progress
            write_progress(progress_row)
            save_json_set(SAVED_ROWS_FILE, saved_rows)
            log(f"📝 Saved rows up to {next_expected - 1}")

        now = time.time()
        if now - last_save_time > FORCE_SAVE_INTERVAL and received:
            merged_rows = []
            for row_idx in sorted(received.keys()):
                if row_idx not in saved_rows:
                    merged_rows.extend(received[row_idx])
                    saved_rows.add(row_idx)
            if merged_rows:
                append_rows(merged_rows)
                save_json_set(SAVED_ROWS_FILE, saved_rows)
                log(f"⚠️ Forced interim save {len(merged_rows)} rows")
            last_save_time = now

    save_json_set(SAVED_ROWS_FILE, saved_rows)
    log("🛑 Writer stopped.")

async def worker(name, semaphore, queue, queue_lock):
    while True:
        async with semaphore:
            async with queue_lock:
                if not queue:
                    return
                task = heapq.heappop(queue)

        b_id, items, attempt = task.batch_id, task.items, task.attempt
        log(f"[{name}] [Batch {b_id}] ({attempt}x) Generating {len(items)} tweets...")

        try:
            mapping = await generate_variants_batch(items)
            batch_items_rows = []
            for row_idx, user_desc, tweet in items:
                variants = mapping[row_idx]
                csv_rows = [[user_desc, tweet, j, v] for j, v in enumerate(variants, start=1)]
                batch_items_rows.append((row_idx, csv_rows))
            await writer_queue.put(("ok", b_id, batch_items_rows))

        except Exception as e:
            log(f"[{name}] ❌ Error batch #{b_id}: {e}")
            await asyncio.sleep(RETRY_DELAY)
            async with queue_lock:
                heapq.heappush(queue, task)

# ========== Main Program ==========
async def main():
    progress = read_progress()
    saved_rows = load_json_set(SAVED_ROWS_FILE)
    log(f"▶️ Resume from progress row {progress}, saved_rows={len(saved_rows)}")

    with open(INPUT_FILE, encoding="utf-8") as fin:
        reader = csv.reader(fin)
        rows = list(reader)
    total = len(rows)
    log(f"📊 Loaded {total} rows")

    queue = []
    batch_id = 0
    current_batch = []
    for i, row in enumerate(rows):
        if i == 0 or i <= progress or len(row) < 2 or i in saved_rows:
            continue
        user_desc, tweet_text = row[0].strip(), row[1].strip()
        if not tweet_text:
            continue
        current_batch.append((i, user_desc, tweet_text))
        if len(current_batch) >= BATCH_SIZE:
            batch_id += 1
            heapq.heappush(queue, BatchTask(batch_id, current_batch[:]))
            current_batch.clear()
        if batch_id * BATCH_SIZE >= LIMIT:
            break
    if current_batch:
        batch_id += 1
        heapq.heappush(queue, BatchTask(batch_id, current_batch[:]))
    log(f"🧾 Enqueued {batch_id} batches")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue_lock = asyncio.Lock()

    writer = asyncio.create_task(writer_task(progress))
    workers = [asyncio.create_task(worker(f"W{n+1}", semaphore, queue, queue_lock))
               for n in range(CONCURRENCY)]
    await asyncio.gather(*workers)

    await writer_queue.put("STOP")
    await writer
    log("🎉 Done!")

if __name__ == "__main__":
    asyncio.run(main())
