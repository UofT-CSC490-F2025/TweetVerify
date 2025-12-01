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
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from accelerate import Accelerator

from peft.tuners.lora import LoraLayer


import modal
import wandb


# ================================================================
# Modal setup
# ================================================================


image = (
    modal.Image.debian_slim()
    .env({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/mnt/cache/hf"
    })
    .pip_install(
        # PyTorch nightlies for CUDA 12.1 + Python 3.12 support
        "torch", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/nightly/cu121"
    )
    .pip_install(
        "transformers", "datasets", "scikit-learn", "pandas", 
        "tqdm", "peft", "tensorboard", "wandb", "accelerate", "matplotlib", "seaborn"
    )
)

app = modal.App("tweetverify-sft-grpo", image=image)
volume = modal.Volume.from_name("tweetverify-model-cache", create_if_missing=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 256
EPOCHS = 100
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
    modules_to_save=["classifier"]
)

# ================================================================
# GRPO config
# ================================================================
class RLCFG:
    group_size = 64
    epochs = 10
    lr = 5e-6
    kl_coef = 0.1


# ================================================================
# Path logic (your original behavior restored)
# ================================================================

def resolve_paths():
    """
    Auto-detect Modal vs Local execution environment.
    FIXED: Now uses env var instead of checking /root existence.
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
        DATA_DIR / "twitter" / "high_quality_human.csv"
    )


def list_checkpoints_local():
    """
    List all saved checkpoint directories under /mnt/cache.
    """
    if not os.path.exists(CKPT_ROOT):
        return []
    items = []
    for name in os.listdir(CKPT_ROOT):
        full = os.path.join(CKPT_ROOT, name)
        if os.path.isdir(full) and ("sft_run" in name or "grpo" in name):
            items.append(full)
    return sorted(items)

def resolve_latest_checkpoint_local():
    """
    Return the most recent checkpoint directory under /mnt/cache.
    """
    ckpts = list_checkpoints_local()
    if not ckpts:
        return None
    return ckpts[-1]


@app.function(
    image=image,
    volumes={"/mnt/cache": volume},
)
def list_checkpoints():
    return list_checkpoints_local()


@app.function(
    image=image,
    volumes={"/mnt/cache": volume},
)
def resolve_latest_checkpoint():
    return resolve_latest_checkpoint_local()


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
        enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = labels
        return enc
    return f

def collate_eval(tokenizer):
    def f(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
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
        base.config.use_cache = False  # avoid memory spikes

        # Enable activation checkpointing (critical for avoiding OOM)
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass  # some models require model.supports_gradient_checkpointing = True

        try:
            base = prepare_model_for_kbit_training(base)
        except Exception:
            pass

        hidden = base.config.hidden_size

        self.base_model = base
        self.classifier = nn.Linear(hidden * 2, 2)

        # PEFT requires the model to expose .config
        self.config = base.config

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Ensure tensors remain on the correct GPU
        device = next(self.parameters()).device

        device = next(self.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
        attention_mask = attention_mask.to(device=device, dtype=torch.long, non_blocking=True)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )


        hidden = outputs.last_hidden_state  # (B, T, H)

        # CLS representation
        cls_rep = hidden[:, 0, :]  # (B, H)

        # Mask-aware mean pooling
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked_hidden = hidden * mask        # (B, T, H)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_rep = masked_hidden.sum(dim=1) / lengths

        pooled = torch.cat([cls_rep, mean_rep], dim=-1)  # (B, 2H)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # always compute loss in fp32 for numerical stability
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 1.5], device=logits.device)
            )
            loss = loss_fn(logits.float(), labels)

        return {"loss": loss, "logits": logits}




# ================================================================
# Snapshot tools
# ================================================================
def score4(m):
    return m["f1"]*0.4 + m["precision"]*0.2 + m["recall"]*0.2 + m["accuracy"]*0.2

def find_top_k(run_dir, k=5):
    out = []
    for d in os.listdir(run_dir):
        mp = os.path.join(run_dir, d, "metrics.json")
        if not os.path.exists(mp): continue
        m = json.load(open(mp))
        out.append((score4(m), d, m))
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:k]


# ================================================================
# GRPO
# ================================================================
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400
)

def compute_rlvr_reward(logits, labels, actions):
    """
    Compute continuous RLVR reward:
    - correctness term
    - logit margin
    - batch F1 shaping
    Followed by normalization (batch-relative soft advantage).
    """
    device = logits.device
    B = labels.size(0)
    idx = torch.arange(B, device=device)

    logp = torch.log_softmax(logits, dim=-1)

    # log p_true minus log p_false = decision margin
    logp_true = logp[idx, labels]
    logp_false = logp[idx, 1 - labels]
    margin = logp_true - logp_false

    # correctness signal
    correct = (actions == labels).float()

    # batch-level classification metrics
    y_true = labels.detach().cpu().numpy()
    y_pred = actions.detach().cpu().numpy()

    try:
        f1 = f1_score(y_true, y_pred)
    except Exception:
        f1 = 0.5

    # continuous reward
    reward_raw = (
        1.0 * correct +
        0.3 * margin +
        0.4 * (f1 - 0.5)
    )

    # normalize reward → soft advantage
    mean = reward_raw.mean()
    std = reward_raw.std().clamp(min=1e-6)
    reward_norm = (reward_raw - mean) / std
    reward_norm = reward_norm.clamp(-5.0, 5.0)

    return reward_norm, {"f1": float(f1)}


@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400
)
def train_grpo(ai_bytes: bytes, human_bytes: bytes, ref_dir: str):

    os.environ["MODAL_ENVIRONMENT"] = "true"

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------
    ai_df = pd.read_csv(io.BytesIO(ai_bytes))
    hu_df = pd.read_csv(io.BytesIO(human_bytes))
    n = min(len(ai_df), len(hu_df))

    ai_df = ai_df.sample(n, random_state=42)
    hu_df = hu_df.sample(n, random_state=42)
    ai_df["label"] = 1
    hu_df["label"] = 0

    df = pd.concat([ai_df, hu_df]).sample(frac=1, random_state=123)

    dataset = Dataset.from_pandas(df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(
        ref_dir, trust_remote_code=True, token=HF_TOKEN
    )
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

    # ----------------------------------------------------------------------
    # SINGLE BACKBONE + two LoRAs: {policy (trainable), ref (frozen)}
    # ----------------------------------------------------------------------

    # 1) Load shared backbone (only ONCE)
    backbone = AutoModel.from_pretrained(
        ref_dir,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto",
        dtype=torch.bfloat16
    )
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable()

    # 2) Wrap backbone into our judging head (same as QwenJudge but without loading twice)
    class SharedQwenJudge(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            hidden = base.config.hidden_size
            self.classifier = nn.Linear(hidden, 2).to(device)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled)
            return {"logits": logits}

    shared = SharedQwenJudge(backbone).to(device)

    # 3) Attach LoRA adapters to the SAME model
    model = get_peft_model(shared, LORA_CFG).to(device)

    # 4) Load SFT LoRA → use as reference adapter
    adapter_dir = os.path.join(ref_dir, "adapters")
    has_sft = os.path.exists(adapter_dir)

    if has_sft:
        model.load_adapter(adapter_dir, "ref")
        print("[init] Loaded SFT LoRA as frozen reference LoRA")
    else:
        print("[warn] No SFT adapters found; reference LoRA will be zero-initialized")
        model.add_adapter("ref", LORA_CFG)

    # 5) Add policy adapter
    model.add_adapter("policy", LORA_CFG)

    # 6) Clone SFT weights: ref → policy
    def clone_lora(src="ref", dst="policy"):
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if src in module.lora_A and dst in module.lora_A:
                    module.lora_A[dst].weight.data.copy_(
                        module.lora_A[src].weight.data
                    )
                    module.lora_B[dst].weight.data.copy_(
                        module.lora_B[src].weight.data
                    )

    if has_sft:
        clone_lora("ref", "policy")
        print("[init] Initialized policy LoRA from SFT snapshot")

    # 7) Freeze reference adapter, enable training for policy adapter
    for module in model.modules():
        if isinstance(module, LoraLayer):
            # ref adapter frozen
            if "ref" in module.lora_A:
                module.lora_A["ref"].weight.requires_grad = False
                module.lora_B["ref"].weight.requires_grad = False
            # policy adapter trainable
            if "policy" in module.lora_A:
                module.lora_A["policy"].weight.requires_grad = True
                module.lora_B["policy"].weight.requires_grad = True

    # By default, activate policy adapter
    model.set_adapter("policy")

    # Optimizer trains ONLY the policy LoRA + classifier
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=RLCFG.lr
    )

    eps_clip = 0.2
    RUN = f"/mnt/cache/grpo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN, exist_ok=True)

    best_f1 = -1.0

    # ----------------------------------------------------------------------
    # GRPO Training Loop
    # ----------------------------------------------------------------------
    for ep in range(1, RLCFG.epochs + 1):

        model.train()
        pbar = tqdm(train_loader, desc=f"[GRPO-RLVR] epoch {ep}")

        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            # ------------------------
            # Policy forward
            # ------------------------
            model.set_adapter("policy")
            out = model(**batch)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)

            idx = torch.arange(actions.size(0), device=device)
            logp = torch.log_softmax(logits, dim=-1)

            # Reward terms
            correct = (actions == batch["labels"]).float()
            logp_true = logp[idx, batch["labels"]]
            logp_false = logp[idx, 1 - batch["labels"]]
            margin = logp_true - logp_false

            y_true_np = batch["labels"].detach().cpu().numpy()
            y_pred_np = actions.detach().cpu().numpy()
            try:
                f1 = f1_score(y_true_np, y_pred_np)
            except:
                f1 = 0.5

            reward = (
                1.0 * correct +
                0.3 * margin +
                0.4 * (f1 - 0.5)
            )

            group_mean = reward.mean().detach()
            advantage = reward - group_mean

            chosen_logp = logp[idx, actions]
            pg_loss = -(advantage * chosen_logp).mean()

            # ------------------------
            # Reference forward (frozen adapter)
            # ------------------------
            with torch.no_grad():
                model.set_adapter("ref")     # switch adapter
                ref_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )["logits"]
                ref_logp = torch.log_softmax(ref_logits, dim=-1)

            # Reset to policy
            model.set_adapter("policy")

            # KL
            kl = (probs * (logp - ref_logp)).sum(dim=-1).mean()

            loss = pg_loss + RLCFG.kl_coef * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "kl": float(kl.detach().cpu()),
            })

            del out, logits, probs, logp, ref_logits, ref_logp
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            model.set_adapter("policy")
            for b in eval_loader:
                for k, v in b.items():
                    if torch.is_tensor(v):
                        b[k] = v.to(device, non_blocking=True)

                o = model(**b)
                preds = o["logits"].argmax(dim=-1)

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(b["labels"].cpu().tolist())

        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_dir = os.path.join(RUN, "best")
            os.makedirs(best_dir, exist_ok=True)
            tokenizer.save_pretrained(best_dir)
            model.save_pretrained(os.path.join(best_dir, "adapters"))

        torch.cuda.empty_cache()

    return {"best_f1": best_f1, "best_checkpoint": best_dir}


# ================================================================
# SFT
# ================================================================
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/mnt/cache": volume},
    timeout=86400
)
def train_sft(ai_bytes: bytes, human_bytes: bytes):

    os.environ["MODAL_ENVIRONMENT"] = "true"

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)
    set_seed(42)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN = f"/mnt/cache/sft_run_{ts}"
    os.makedirs(RUN, exist_ok=True)

    # -----------------------------
    # Data
    # -----------------------------
    ai = pd.read_csv(io.BytesIO(ai_bytes))
    hu = pd.read_csv(io.BytesIO(human_bytes))
    n = min(len(ai), len(hu))
    ai = ai.sample(n)
    hu = hu.sample(n)
    ai["label"] = 1
    hu["label"] = 0
    df = pd.concat([ai, hu]).sample(frac=1, random_state=999)

    ds = Dataset.from_pandas(df[["text", "label"]])
    split = ds.train_test_split(test_size=0.1, seed=999)
    tr, ev = split["train"], split["test"]


    train_loader = DataLoader(tr, batch_size=24, shuffle=True, collate_fn=collate_train(tokenizer))
    eval_loader  = DataLoader(ev, batch_size=16, shuffle=False, collate_fn=collate_eval(tokenizer))

    # ===========================================================
    # SFT Warm-Start
    # ===========================================================

    # 1. Load tokenizer
    if ref_dir and os.path.exists(ref_dir):
        print(f"[SFT] Warm-start tokenizer from {ref_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            ref_dir, trust_remote_code=True, token=HF_TOKEN
        )
    else:
        print("[SFT] No previous tokenizer found — using BASE_MODEL tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=HF_TOKEN
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


    # 2. Backbone: Always from BASE_MODEL (same as GRPO)
    print("[SFT] Loading BASE_MODEL backbone (backbone is never saved)")
    base = QwenJudge()   # Qwen backbone + classifier head


    # 3. Wrap with PEFT
    model = get_peft_model(base, LORA_CFG)


    # 4. Load LoRA adapter if exists (same as GRPO)
    adapter_dir = os.path.join(ref_dir, "adapters") if ref_dir else None

    if adapter_dir and os.path.exists(adapter_dir):
        print(f"[SFT] Warm-start LoRA adapter from {adapter_dir}")
        model.load_adapter(adapter_dir, adapter_name="default")
        model.set_adapter("default")
    else:
        print("[SFT] No previous LoRA adapters found — training LoRA from scratch.")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-6)

    model, opt, train_loader, eval_loader = accelerator.prepare(
        model, opt, train_loader, eval_loader
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, 0, total_steps)

    # -----------------------------
    # Training
    # -----------------------------
    for ep in range(1, EPOCHS+1):

        model.train()
        for batch in tqdm(train_loader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out["loss"]
                accelerator.backward(loss)
                opt.step()
                scheduler.step()
                opt.zero_grad()

        # -----------------------------
        # Eval
        # -----------------------------
        if accelerator.is_main_process:
            model.eval()
            YT, YP = [], []
            with torch.no_grad():
                for b in eval_loader:
                    out = model(**b)
                    pred = out["logits"].argmax(-1)
                    YP += pred.cpu().tolist()
                    YT += b["labels"].cpu().tolist()

            acc = accuracy_score(YT, YP)
            pr, re, f1v, _ = precision_recall_fscore_support(YT, YP, average="binary")
            score = score4({"accuracy": acc, "precision": pr, "recall": re, "f1": f1v})

            ep_dir = os.path.join(RUN, f"epoch_{ep}_f1-{f1v:.4f}_p-{pr:.4f}")
            os.makedirs(ep_dir, exist_ok=True)
            tokenizer.save_pretrained(ep_dir)
            accelerator.unwrap_model(model).save_pretrained(os.path.join(ep_dir, "adapters"))
            json.dump({"accuracy":acc,"precision":pr,"recall":re,"f1":f1v},
                      open(os.path.join(ep_dir, "metrics.json"), "w"), indent=2)

    # -----------------------------
    # Select top-1 snapshot and run GRPO
    # -----------------------------
    if accelerator.is_main_process:
        top5 = find_top_k(RUN, k=5)
        best = top5[0]
        best_dir = os.path.join(RUN, best[1])

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

    Usage examples:
        modal run train_judge_qwen --cmd train
        modal run train_judge_qwen --cmd grpo
    """

    # Resolve local dataset paths
    AI_LOCAL, HUMAN_LOCAL = resolve_paths()
    print(f"AI dataset: {AI_LOCAL}")
    print(f"Human dataset: {HUMAN_LOCAL}")

    # ------------------------------------------------------------
    # TRAIN (SFT stage)
    # ------------------------------------------------------------
    if cmd == "sft":
        # Optional warm start from latest checkpoint
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if ref_dir:
            print(f"ℹ️ Using existing checkpoint for warm start: {ref_dir}")
        else:
            print("⚠️ No previous checkpoint found — SFT will start from BASE_MODEL.")

        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            result = train_sft.remote(f1.read(), f2.read())
            print("✅ SFT job finished:", result)

    # ------------------------------------------------------------
    # GRPO stage
    # ------------------------------------------------------------
    elif cmd == "grpo":
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if not ref_dir:
            print("❌ No checkpoint found. Please run SFT first or specify --ref_best_dir.")
            return

        print(f"Using reference checkpoint: {ref_dir}")

        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            result = train_grpo.remote(f1.read(), f2.read(), ref_dir)
            print("✅ GRPO job finished:", result)