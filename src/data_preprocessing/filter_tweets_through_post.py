import asyncio
import csv
import heapq
import json
import os
import re
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List

import tiktoken
from openai import AsyncOpenAI
from openai._exceptions import APIError, APIConnectionError, RateLimitError

# ========== 基础配置 ==========
API_KEY = os.getenv("OPEN_AI_API_KEY")

ROOT = Path(__file__).resolve().parent.parent

folder = ROOT / "datalake" / "dataset" / "Political" / "Global Political tweets"

MODEL = "gpt-4o-mini"
FOLDER = r"E:/程序/CSC490/src/dataset/Political/Global Political tweets/"
INPUT_FILE = folder / "cleaned_output.csv"
OUTPUT_FILE = folder / "good_tweets_async.csv"
PROGRESS_FILE = folder / "progress_filter.txt"
STATUS_LOG = folder / "async_filter_status.log"

# ========== 参数 ==========
LIMIT = 100000
CONCURRENCY = 50
SAVE_INTERVAL = 20  # 每多少 GOOD 推文写入一次
RETRY_DELAY = 2
RATE_LIMIT = 400
WINDOW = 60
TARGET_TOKENS = 1100  # 每批最少 token 数

client = AsyncOpenAI(api_key=API_KEY)
enc = tiktoken.get_encoding("cl100k_base")
writer_queue = asyncio.Queue()
processed_ref = []

QUALITY_PROMPT = (
    "You are a tweet quality classifier.\n\n"
    "For each tweet in the list below, decide if it is a high-quality, politically relevant, English-language tweet "
    "that expresses a clear political viewpoint, opinion, or reaction to current events.\n\n"
    "You MUST output a valid JSON object following **exactly this format**:\n\n"
    "{\n"
    "  \"results\": [\n"
    "    {\"tweet\": \"<original tweet 1>\", \"label\": \"GOOD\"},\n"
    "    {\"tweet\": \"<original tweet 2>\", \"label\": \"BAD\"}\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Only output valid JSON (no comments or explanations).\n"
    "- Each tweet must appear exactly once.\n"
    "- Label must be either GOOD or BAD (uppercase).\n"
    "- No trailing commas or text outside JSON.\n\n"
    "Now classify the following tweets:\n"
)

# ========== 工具函数 ==========
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def append_rows(rows):
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(["user_name", "tweet"])
        writer.writerows(rows)

def write_progress(n):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(n))

def read_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            return int(open(PROGRESS_FILE, "r").read().strip())
        except:
            return 0
    return 0

def estimate_tokens(tweets: List[str]) -> int:
    joined_tweets = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweets)])
    text = QUALITY_PROMPT + joined_tweets
    return len(enc.encode(text))

# ========== Writer 协程（唯一写入点）==========
async def writer_task(processed_ref):
    log("🟢 Writer started and waiting for data...")
    pending = []
    last_save_time = time.time()
    while True:
        item = await writer_queue.get()
        if item == "STOP":
            break
        _, batch_idx, rows = item
        pending.extend(rows)

        # SAVE_INTERVAL 条或间隔超过 30 秒写一次
        if len(pending) >= SAVE_INTERVAL or (time.time() - last_save_time > 30):
            append_rows(pending)
            write_progress(processed_ref[0])
            log(f"📝 Writer saved {len(pending)} GOOD tweets (up to batch {batch_idx})")
            pending.clear()
            last_save_time = time.time()

    if pending:
        append_rows(pending)
        log(f"📝 Writer final save {len(pending)} GOOD tweets")

# ========== 限速 ==========
timestamps = deque()
async def rate_limit_guard():
    now = time.time()
    timestamps.append(now)
    while timestamps and timestamps[0] < now - WINDOW:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT:
        sleep_time = WINDOW - (now - timestamps[0]) + 0.1
        log(f"⏳ Throttling {sleep_time:.2f}s to stay under {RATE_LIMIT} RPM...")
        await asyncio.sleep(sleep_time)

# ========== 任务定义 ==========
class Task:
    def __init__(self, priority, idx, batch, usernames, attempt=1):
        self.priority = priority
        self.idx = idx
        self.batch = batch
        self.usernames = usernames
        self.attempt = attempt
    def __lt__(self, other):
        return self.priority < other.priority

# ========== 分类函数 ==========
async def classify_batch(task: Task):
    await rate_limit_guard()
    joined_tweets = "\n".join([f"{i+1}. {t}" for i, t in enumerate(task.batch)])
    prompt = QUALITY_PROMPT + joined_tweets

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
    )

    usage = getattr(resp, "usage", None)
    if usage:
        cached = getattr(usage, "prompt_tokens_details", None)
        if cached:
            cached_in = getattr(cached, "cached_tokens", 0)
        else:
            cached_in = getattr(usage, "cached_input_tokens", 0) or getattr(usage, "input_cached_tokens", 0)
        total_in = getattr(usage, "prompt_tokens", 0)
        hit_rate = (cached_in / total_in * 100) if total_in else 0
        log(f"[CACHE] batch={task.idx}   ({hit_rate:.1f}% hit)  {usage}")

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        results = data.get("results", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            log(f"⚠️ JSON parse failed (no braces) at batch {task.idx}")
            return [(u, t, "BAD") for u, t in zip(task.usernames, task.batch)]
        try:
            data = json.loads(m.group())
            results = data.get("results", [])
        except Exception as e:
            log(f"⚠️ JSON parse failed batch {task.idx}: {e}")
            return [(u, t, "BAD") for u, t in zip(task.usernames, task.batch)]

    # ✅ 一致性检查：长度对齐（缺的补 BAD，多的截断）
    if len(results) != len(task.batch):
        log(f"⚠️ Batch {task.idx}: expected {len(task.batch)} results, got {len(results)} — fixing by index")
        fixed = []
        for i in range(len(task.batch)):
            if i < len(results) and isinstance(results[i], dict):
                fixed.append(results[i])
            else:
                fixed.append({"tweet": task.batch[i], "label": "BAD"})
        results = fixed

    # ✅ 索引对齐取 label（不要再按 tweet 字符串匹配）
    labels = []
    for i in range(len(task.batch)):
        r = results[i] if i < len(results) else {}
        lbl = (r.get("label") or "").upper()
        if lbl not in ("GOOD", "BAD"):
            lbl = "BAD"
        labels.append(lbl)

    return list(zip(task.usernames, task.batch, labels))

    return list(zip(task.usernames, task.batch, labels))

# ========== 工作者 ==========
async def worker(name, semaphore, queue, processed_ref, lock):
    while True:
        async with semaphore:
            if not queue:
                return
            task = heapq.heappop(queue)
        log(f"[{name}] [Batch {task.idx}] ({task.attempt}x) Processing...")

        try:
            rows = await classify_batch(task)
            good_rows = [(u, t) for (u, t, label) in rows if label == "GOOD"]

            async with lock:
                processed_ref[0] += len(task.batch)
                count = processed_ref[0]

            if good_rows:
                await writer_queue.put(("batch", task.idx, good_rows))

            if count % 10 == 0:
                log(f"[{name}] ✅ {count} tweets processed")

        except RateLimitError:
            wait = RETRY_DELAY * (2 ** (task.attempt - 1))
            heapq.heappush(queue, Task(priority=time.time()+wait, idx=task.idx,
                                       batch=task.batch, usernames=task.usernames,
                                       attempt=task.attempt+1))
            log(f"[{name}] ⚠️ RateLimit on batch #{task.idx}, retry in {wait:.1f}s")
            await asyncio.sleep(wait)
        except (APIError, APIConnectionError, asyncio.TimeoutError) as e:
            log(f"[{name}] ⚠️ Transient error on batch #{task.idx}: {e}")
            await asyncio.sleep(RETRY_DELAY)
            heapq.heappush(queue, task)
        except Exception as e:
            log(f"[{name}] ❌ Fatal error on batch #{task.idx}: {e}")
            log(traceback.format_exc())

# ========== 主流程 ==========
async def main():
    last_progress = read_progress()
    log(f"▶️ Resuming from batch {last_progress}")

    queue = []
    with open(INPUT_FILE, encoding="utf-8", errors="ignore") as fin:
        reader = csv.reader(fin)
        next(reader, None)
        tweets, names = [], []
        batch_id = 0
        total_tokens = 0

        for i, row in enumerate(reader, start=1):
            if i <= last_progress or len(row) < 2:
                continue
            username, text = row[0].strip(), row[1].strip()
            if not text:
                continue
            tweets.append(text)
            names.append(username)
            total_tokens = estimate_tokens(tweets)

            if total_tokens >= TARGET_TOKENS:
                batch_id += 1
                heapq.heappush(queue, Task(priority=batch_id, idx=batch_id, batch=tweets[:], usernames=names[:]))
                log(f"📦 Created batch #{batch_id} ({len(tweets)} tweets, ~{total_tokens} tokens)")
                tweets.clear()
                names.clear()
                total_tokens = 0

            if batch_id * 10 >= LIMIT:
                break

        if tweets:
            batch_id += 1
            heapq.heappush(queue, Task(priority=batch_id, idx=batch_id, batch=tweets[:], usernames=names[:]))
            log(f"📦 Created final batch #{batch_id} ({len(tweets)} tweets, ~{total_tokens} tokens)")

    log(f"📊 Loaded {len(queue)} adaptive batches (≥{TARGET_TOKENS} tokens each)")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    processed_ref = [last_progress]
    lock = asyncio.Lock()

    writer = asyncio.create_task(writer_task(processed_ref))
    workers = [asyncio.create_task(worker(f"W{n+1}", semaphore, queue, processed_ref, lock))
               for n in range(CONCURRENCY)]
    await asyncio.gather(*workers)

    await writer_queue.put("STOP")
    await writer

    write_progress(processed_ref[0])
    log("🎉 Done!")

if __name__ == "__main__":
    asyncio.run(main())
