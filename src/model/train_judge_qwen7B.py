"""
FINAL VERSION — Qwen2.5-7B SFT + GRPO Pipeline
Needs to setup Modal api key beforehand and 
Needs the following api keys to run:
"HF_TOKEN" and "WANDB_API_KEY"

then navigate to src/model and run:
modal run -m train_judge_qwen7B --cmd sft
modal run -m train_judge_qwen7B --cmd grpo
"""

import os, io, json, torch, pandas as pd, numpy as np, random

from pathlib import Path
from datetime import datetime

from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
)
from accelerate import Accelerator
from peft.tuners.lora import LoraLayer

import modal
import wandb


# ================================================================
# Modal setup
# ================================================================

image = (
    modal.Image.debian_slim()
    .env(
        {
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": "/mnt/cache/hf",
        }
    )
    .pip_install(
        # PyTorch nightlies for CUDA 12.1 + Python 3.12 support
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/nightly/cu121",
    )
    .pip_install(
        "transformers",
        "datasets",
        "scikit-learn",
        "pandas",
        "tqdm",
        "peft",
        "tensorboard",
        "wandb",
        "accelerate",
        "matplotlib",
        "seaborn",
    )
)

app = modal.App("tweetverify-sft-grpo", image=image)
volume = modal.Volume.from_name("tweetverify-model-cache", create_if_missing=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 256
EPOCHS = 6
CKPT_ROOT = "/mnt/cache"


# ================================================================
# LoRA config
# ================================================================
LORA_CFG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier"],
)

# ================================================================
# GRPO config
# ================================================================
class RLCFG:
    group_size = 64
    epochs = 6
    lr = 5e-6
    kl_coef = 0.1


# ================================================================
# Path logic
# ================================================================


def resolve_paths():
    """
    Auto-detect Modal vs Local execution environment.
    """

    IS_MODAL = os.environ.get("MODAL_ENVIRONMENT") == "true"

    if IS_MODAL:
        DATA_DIR = Path("/root/data")
        return DATA_DIR / "ai_generated.csv", DATA_DIR / "high_quality_human.csv"

    # Local mode
    PROJECT_DIR = Path(__file__).resolve()
    for _ in range(5):
        if (PROJECT_DIR / "datalake").exists():
            break
        PROJECT_DIR = PROJECT_DIR.parent

    DATA_DIR = PROJECT_DIR / "datalake" / "curated"
    return (
        DATA_DIR / "llm" / "ai_generated.csv",
        DATA_DIR / "twitter" / "high_quality_human.csv",
    )


def get_latest_checkpoint(root="/mnt/cache"):
    """
    Find the most recent *usable* checkpoint directory (SFT or GRPO) under /mnt/cache.

    Strategy:
    1. Enumerate all sft_run_* / grpo_run_* directories, sorted by mtime from newest to oldest;
    2. For each run:
       - Prefer using run/best (if exists);
       - Otherwise, look for epoch_* directories and take the last epoch;
       - If the run has neither best nor epoch_*, skip to the next run;
    3. If no runs have best/epoch_*, return None.
    """

    if not os.path.isdir(root):
        print(f"[warn] Cache directory {root} not found.")
        return None

    runs = []
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if os.path.isdir(full) and (d.startswith("sft_run_") or d.startswith("grpo_run_")):
            mtime = os.path.getmtime(full)
            runs.append((mtime, full))

    if not runs:
        print("❌ No SFT or GRPO checkpoint runs found under", root)
        return None

    runs.sort(key=lambda x: x[0], reverse=True)

    for _, run_dir in runs:
        print(f"[check] inspecting run: {run_dir}")

        best_dir = os.path.join(run_dir, "best")
        if os.path.isdir(best_dir):
            print(f"[auto] Using BEST checkpoint: {best_dir}")
            return best_dir

        epoch_dirs = [
            os.path.join(run_dir, d)
            for d in os.listdir(run_dir)
            if d.startswith("epoch_") and os.path.isdir(os.path.join(run_dir, d))
        ]

        if epoch_dirs:
            epoch_dirs.sort()
            chosen = epoch_dirs[-1]
            print(f"[auto] Using EPOCH checkpoint: {chosen}")
            return chosen

        print(f"[skip] Run {run_dir} has no best/epoch_* — checking older runs...")

    print("⚠️ No usable checkpoints found — will fall back to BASE_MODEL")
    return None


@app.function(image=image, volumes={"/mnt/cache": volume})
def resolve_latest_checkpoint():
    return get_latest_checkpoint("/mnt/cache")


# ================================================================
# Utils
# ================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_train(tokenizer):
    def f(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return f


def collate_eval(tokenizer):
    def f(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return f


# ================================================================
# Model
# ================================================================
class QwenJudge(nn.Module):
    """
    Hybrid pooled classifier on top of Qwen2.5.
    Includes:
    - CLS token representation
    - Mask-aware mean pooling
    - Activation checkpointing for memory efficiency
    """

    def __init__(self):
        super().__init__()

        base = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        base.config.use_cache = False

        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass

        try:
            base = prepare_model_for_kbit_training(base)
        except Exception:
            pass

        hidden = base.config.hidden_size

        self.base_model = base
        self.classifier = nn.Linear(hidden * 2, 2)
        base_dtype = next(self.base_model.parameters()).dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

        self.config = base.config

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
        attention_mask = attention_mask.to(device=device, dtype=torch.long, non_blocking=True)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden = outputs.last_hidden_state  # (B, T, H)

        cls_rep = hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_rep = masked_hidden.sum(dim=1) / lengths

        pooled = torch.cat([cls_rep, mean_rep], dim=-1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 1.5], device=logits.device)
            )
            loss = loss_fn(logits.float(), labels)

        return {"loss": loss, "logits": logits}


# ================================================================
# Snapshot tools
# ================================================================
def score4(m):
    return m["f1"] * 0.4 + m["precision"] * 0.2 + m["recall"] * 0.2 + m["accuracy"] * 0.2


def find_top_k(run_dir, k=5):
    out = []
    for d in os.listdir(run_dir):
        mp = os.path.join(run_dir, d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        out.append((score4(m), d, m))
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:k]


# ================================================================
# GRPO reward (helper, not used directly in loop)
# ================================================================
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400,
)
def compute_rlvr_reward(logits, labels, actions):
    device = logits.device
    B = labels.size(0)
    idx = torch.arange(B, device=device)

    logp = torch.log_softmax(logits, dim=-1)

    logp_true = logp[idx, labels]
    logp_false = logp[idx, 1 - labels]
    margin = logp_true - logp_false

    correct = (actions == labels).float()

    y_true = labels.detach().cpu().numpy()
    y_pred = actions.detach().cpu().numpy()

    try:
        f1 = f1_score(y_true, y_pred)
    except Exception:
        f1 = 0.5

    reward_raw = 1.0 * correct + 0.3 * margin + 0.4 * (f1 - 0.5)

    mean = reward_raw.mean()
    std = reward_raw.std().clamp(min=1e-6)
    reward_norm = (reward_raw - mean) / std
    reward_norm = reward_norm.clamp(-5.0, 5.0)

    return reward_norm, {"f1": float(f1)}


# ================================================================
# GRPO
# ================================================================
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400,
)
def train_grpo(ai_bytes: bytes, human_bytes: bytes, ref_dir: str):

    os.environ["MODAL_ENVIRONMENT"] = "true"

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---------------------- Dataset ---------------------------
    ai_df = pd.read_csv(io.BytesIO(ai_bytes))
    hu_df = pd.read_csv(io.BytesIO(human_bytes))
    n = min(len(ai_df), len(hu_df))

    ai_df = ai_df.sample(n, random_state=42)
    hu_df = hu_df.sample(n, random_state=42)
    ai_df["label"] = 1
    hu_df["label"] = 0

    df = pd.concat([ai_df, hu_df]).sample(frac=1, random_state=123)
    dataset = Dataset.from_pandas(df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(ref_dir, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    split = dataset.train_test_split(test_size=0.1, seed=42)
    tr, ev = split["train"], split["test"]

    train_loader = DataLoader(
        tr,
        batch_size=RLCFG.group_size,
        shuffle=True,
        collate_fn=collate_train(tokenizer),
    )
    eval_loader = DataLoader(
        ev,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_eval(tokenizer),
    )

    # ---------------------- Backbone --------------------------
    backbone = AutoModel.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map=None,
        dtype=torch.bfloat16,
    )
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable()
    
    try:
        backbone = prepare_model_for_kbit_training(backbone)
    except:
        pass

    class SharedQwenJudge(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            hidden = base.config.hidden_size
            self.config = base.config

            # Exactly consistent with SFT: 2H -> 2
            self.classifier = nn.Linear(hidden * 2, 2)

            # Align classifier dtype with backbone (will align again later)
            base_dtype = next(self.base.parameters()).dtype
            self.classifier = self.classifier.to(dtype=base_dtype)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = out.last_hidden_state  # (B, T, H)

            # CLS + mask-aware mean pooling, fully aligned with QwenJudge
            cls_rep = hidden[:, 0, :]                       # (B, H)
            mask = attention_mask.unsqueeze(-1)             # (B, T, 1)
            masked_hidden = hidden * mask                   # (B, T, H)
            lengths = mask.sum(dim=1).clamp(min=1)          # (B, 1)
            mean_rep = masked_hidden.sum(dim=1) / lengths   # (B, H)

            pooled = torch.cat([cls_rep, mean_rep], dim=-1)  # (B, 2H)
            logits = self.classifier(pooled)
            return {"logits": logits}



    shared = SharedQwenJudge(backbone).to(device)

    # ---------------------- PEFT model ------------------------
    model = get_peft_model(shared, LORA_CFG).to(device)

    # ---------------------- LoRA warm start -------------------
    has_sft = False

    if ref_dir != "BASE":
        if ref_dir is None:
            ref_dir = get_latest_checkpoint(CKPT_ROOT)

        adapter_dir = os.path.join(ref_dir, "adapters") if ref_dir else None

        print("[GRPO] ref_dir:", ref_dir)
        print("[GRPO] adapter_dir:", adapter_dir)
        if adapter_dir:
            print("[GRPO] exists:", os.path.exists(os.path.join(adapter_dir, "adapter_config.json")))
        else:
            print("[GRPO] exists: False (no adapter_dir)")

        if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            print(f"[GRPO] Loading SFT LoRA as 'ref' from {adapter_dir}")
            model.load_adapter(adapter_dir, adapter_name="ref")
            has_sft = True
        else:
            raise RuntimeError(
                f"[GRPO ERROR] Expected SFT adapter under {adapter_dir}, "
                "but adapter_config.json was not found. "
                "GRPO requires a valid SFT checkpoint as reference."
            )
    else:
        raise RuntimeError(
            "[GRPO ERROR] ref_dir='BASE' is not allowed. "
            "GRPO requires a reference model (SFT LoRA). "
            "You must run SFT first to generate a checkpoint."
        )

    # ---------------------- policy adapter --------------------
    print("[GRPO] Creating 'policy' adapter")
    model.add_adapter("policy", LORA_CFG)

    if has_sft:
        print("[GRPO] Initializing 'policy' from 'ref'")
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if "ref" in module.lora_A and "policy" in module.lora_A:
                    module.lora_A["policy"].weight.data.copy_(
                        module.lora_A["ref"].weight.data
                    )
                    module.lora_B["policy"].weight.data.copy_(
                        module.lora_B["ref"].weight.data
                    )
    
    target_dtype = next(model.base.parameters()).dtype if hasattr(model, "base") else next(model.parameters()).dtype
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.to(dtype=target_dtype)

    for module in model.modules():
        if isinstance(module, LoraLayer):
            if "ref" in module.lora_A:
                module.lora_A["ref"].weight.requires_grad = False
                module.lora_B["ref"].weight.requires_grad = False
            if "policy" in module.lora_A:
                module.lora_A["policy"].weight.requires_grad = True
                module.lora_B["policy"].weight.requires_grad = True

    model.set_adapter("policy")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=RLCFG.lr,
    )

    RUN = f"/mnt/cache/grpo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN, exist_ok=True)

    best_f1 = -1.0
    best_dir = None

    # ---------------------- GRPO loop -------------------------
    for ep in range(1, RLCFG.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[GRPO-RLVR] epoch {ep}")

        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            # Policy
            model.set_adapter("policy")
            out = model(**batch)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)

            idx = torch.arange(actions.size(0), device=device)
            logp = torch.log_softmax(logits, dim=-1)

            correct = (actions == batch["labels"]).float()
            logp_true = logp[idx, batch["labels"]]
            logp_false = logp[idx, 1 - batch["labels"]]
            margin = logp_true - logp_false

            y_true_np = batch["labels"].detach().cpu().numpy()
            y_pred_np = actions.detach().cpu().numpy()
            try:
                f1 = f1_score(y_true_np, y_pred_np)
            except Exception:
                f1 = 0.5

            reward = 1.0 * correct + 0.3 * margin + 0.4 * (f1 - 0.5)
            group_mean = reward.mean().detach()
            advantage = reward - group_mean

            chosen_logp = logp[idx, actions]
            pg_loss = -(advantage * chosen_logp).mean()

            # Reference
            with torch.no_grad():
                model.set_adapter("ref")
                ref_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )["logits"]
                ref_logp = torch.log_softmax(ref_logits, dim=-1)

            model.set_adapter("policy")

            kl = (probs * (logp - ref_logp)).sum(dim=-1).mean()

            loss = pg_loss + RLCFG.kl_coef * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            pbar.set_postfix(
                {
                    "loss": float(loss.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                }
            )

            del out, logits, probs, logp, ref_logits, ref_logp
            torch.cuda.empty_cache()

        # ---------------------- Eval ---------------------------
        model.eval()
        y_true, y_pred = [], []
        y_score = [] 

        with torch.no_grad():
            model.set_adapter("policy")
            for b in eval_loader:
                for k, v in b.items():
                    if torch.is_tensor(v):
                        b[k] = v.to(device, non_blocking=True)

                o = model(**b)
                logits = o["logits"]
                preds = logits.argmax(dim=-1)

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(b["labels"].cpu().tolist())
                
                probs = torch.softmax(logits, dim=-1)[:, 1]
                y_score.extend(probs.cpu().tolist())

        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        acc = accuracy_score(y_true, y_pred)
        pr, re, f1_eval, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        try:
            auc_eval = roc_auc_score(y_true, y_score)
        except Exception:
            auc_eval = 0.5
        
        print(f"[GRPO] epoch {ep}  acc={acc:.4f}  pr={pr:.4f}  re={re:.4f}  f1={f1_eval:.4f}  auc={auc_eval:.4f}")

        if f1_eval > best_f1:
            best_f1 = f1_eval
            best_dir = os.path.join(RUN, "best")
            os.makedirs(best_dir, exist_ok=True)
            tokenizer.save_pretrained(best_dir)
            model.set_adapter("policy")
            model.save_pretrained(os.path.join(best_dir, "adapters"))
            json.dump(
                {
                    "accuracy": acc,
                    "precision": pr,
                    "recall": re,
                    "f1": f1_eval,
                    "auc": auc_eval,
                },
                open(os.path.join(best_dir, "metrics.json"), "w"),
                indent=2,
            )

        torch.cuda.empty_cache()

    return {"best_f1": best_f1, "best_checkpoint": best_dir}


# ================================================================
# SFT
# ================================================================
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400,
)
def train_sft(ai_bytes: bytes, human_bytes: bytes, ref_dir: str = None):
    os.environ["MODAL_ENVIRONMENT"] = "true"

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)
    set_seed(42)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN = f"/mnt/cache/sft_run_{ts}"
    os.makedirs(RUN, exist_ok=True)

    # ---------------------- Data ------------------------------
    ai = pd.read_csv(io.BytesIO(ai_bytes))
    hu = pd.read_csv(io.BytesIO(human_bytes))
    n = min(len(ai), len(hu))
    ai = ai.sample(n, random_state=42)
    hu = hu.sample(n, random_state=42)
    ai["label"] = 1
    hu["label"] = 0
    df = pd.concat([ai, hu]).sample(frac=1, random_state=999)

    ds = Dataset.from_pandas(df[["text", "label"]])
    split = ds.train_test_split(test_size=0.1, seed=999)
    tr, ev = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        tr, batch_size=24, shuffle=True, collate_fn=collate_train(tokenizer)
    )
    eval_loader = DataLoader(
        ev, batch_size=16, shuffle=False, collate_fn=collate_eval(tokenizer)
    )

    print("[SFT] Loading BASE_MODEL backbone (backbone is never saved)")
    base = QwenJudge()

    # ---------------------- LoRA warm start -------------------
    if ref_dir != "BASE":
        if ref_dir is None:
            ref_dir = get_latest_checkpoint(CKPT_ROOT)

        adapter_dir = os.path.join(ref_dir, "adapters") if ref_dir else None

        print("ref_dir:", ref_dir)
        print("adapter_dir:", adapter_dir)
        if adapter_dir:
            print("exists:", os.path.exists(os.path.join(adapter_dir, "adapter_config.json")))
        else:
            print("exists: False (no adapter_dir)")

        if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            print(f"[SFT] Warm-start LoRA adapter from {adapter_dir}")

            model = get_peft_model(base, LORA_CFG)

            model.load_adapter(adapter_dir, adapter_name="default")

            model.set_adapter("default")

        else:
            print("[SFT] No previous LoRA adapters found — training LoRA from scratch.")
            model = get_peft_model(base, LORA_CFG)
    else:
        print("[SFT] ref=BASE — training LoRA from scratch.")
        model = get_peft_model(base, LORA_CFG)

    # ---------------------- Optimizer & Accelerator -----------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-6)

    model, opt, train_loader, eval_loader = accelerator.prepare(
        model, opt, train_loader, eval_loader
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, 0, total_steps)

    # ---------------------- Training loop ---------------------
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in tqdm(
            train_loader, disable=not accelerator.is_main_process, desc=f"[SFT] epoch {ep}"
        ):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out["loss"]
                accelerator.backward(loss)
                opt.step()
                scheduler.step()
                opt.zero_grad()

        # ------------------ Eval per epoch ---------------------
        if accelerator.is_main_process:
            model.eval()
            YT, YP = [], []
            YS = []  # soft scores
            with torch.no_grad():
                for b in eval_loader:
                    out = model(**b)
                    pred = out["logits"].argmax(-1)
                    YP += pred.cpu().tolist()
                    YT += b["labels"].cpu().tolist()
                    probs = torch.softmax(out["logits"], dim=-1)[:, 1]
                    YS += probs.cpu().tolist()

            acc = accuracy_score(YT, YP)
            pr, re, f1v, _ = precision_recall_fscore_support(
                YT, YP, average="binary"
            )
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(YT, YS)
            except:
                auc = 0.5

            ep_dir = os.path.join(RUN, f"epoch_{ep}_f1-{f1v:.4f}_p-{pr:.4f}")
            os.makedirs(ep_dir, exist_ok=True)
            tokenizer.save_pretrained(ep_dir)

            # unwrap to save only underlying PEFT model (LoRA adapter)
            peft_model = accelerator.unwrap_model(model)
            peft_model.save_pretrained(os.path.join(ep_dir, "adapters"))

            json.dump(
                {"accuracy": acc, "precision": pr, "recall": re, "f1": f1v, "auc": auc},
                open(os.path.join(ep_dir, "metrics.json"), "w"),
                indent=2,
            )

    # ---------------------- After SFT → trigger GRPO ----------
    if accelerator.is_main_process:
        top5 = find_top_k(RUN, k=5)
        best = top5[0]
        best_dir = os.path.join(RUN, best[1])
        print(f"[SFT] Best epoch checkpoint: {best_dir}")

        train_grpo.remote(ai_bytes, human_bytes, best_dir)

    return {"run_dir": RUN}


# ================================================================
# Entry point
# ================================================================
@app.local_entrypoint()
def main(
    cmd: str = "sft",
    ref_best_dir: str = None,
    checkpoint_name: str = None,
):
    """
    Entry point for controlling SFT, GRPO, checkpoint management.
    """

    AI_LOCAL, HUMAN_LOCAL = resolve_paths()
    print(f"AI dataset: {AI_LOCAL}")
    print(f"Human dataset: {HUMAN_LOCAL}")

    if cmd == "sft":
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if ref_dir:
            print(f"ℹ️ Using existing checkpoint for warm start: {ref_dir}")
        else:
            print("⚠️ No previous checkpoint found — SFT will start from BASE_MODEL.")

        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            result = train_sft.remote(f1.read(), f2.read(), ref_dir)
            print("✅ SFT job submitted:", result)

    elif cmd == "grpo":
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if not ref_dir:
            print("❌ No checkpoint found. Please run SFT first or specify --ref_best_dir.")
            return

        print(f"Using reference checkpoint: {ref_dir}")

        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            result = train_grpo.remote(f1.read(), f2.read(), ref_dir)
            print("✅ GRPO job submitted:", result)
