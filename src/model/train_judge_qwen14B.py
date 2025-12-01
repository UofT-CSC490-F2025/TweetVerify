"""
Supervised fine-tuning (SFT) stage with LoRA upon Qwen2.5-14B, TensorBoard/W&B logging,
early-stopping on F1, safe auto-trigger to GRPO using Modal API,
and integrated checkpoint management (list + rollback).
(DDP multi-GPU via HuggingFace Accelerate)

Needs to setup Modal api key beforehand and 
Needs the following api keys to run:
"HF_TOKEN" and "WANDB_API_KEY"

then navigate to src/model and run:
modal run -m train_judge_qwen14B --cmd sft
modal run -m train_judge_qwen14B --cmd grpo
"""

import os, io, json, logging, torch, pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import modal


from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import wandb, uuid
from pathlib import Path



# ==== Modal config ====
image = (
    modal.Image.debian_slim()
    .env({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),  # both names for safety
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "HF_HOME": "/mnt/cache/hf",  # store cache in the mounted volume
    })
    .pip_install(
        "torch", "transformers", "datasets", "scikit-learn", "pandas",
        "tqdm", "peft", "tensorboard", "wandb", "accelerate", "matplotlib", "seaborn"
    )
)

app = modal.App("tweetverify-sft-autogrpo", image=image)
volume = modal.Volume.from_name("tweetverify-model-cache", create_if_missing=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "<HF_TOKEN>")
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
CACHE_ROOT = "/mnt/cache"
# ==== Path setup ====


IS_MODAL = os.path.exists("/root") and not Path("/root").joinpath("datalake").exists()

if IS_MODAL:
    # Mount target (Option 2): only CSVs mounted under /root/data
    PROJECT_ROOT = Path("/root")
    DATA_DIR = PROJECT_ROOT / "data"

    # Mounted files on Modal container
    AI_LOCAL = DATA_DIR / "ai_generated.csv"
    HUMAN_LOCAL = DATA_DIR / "high_quality_human.csv"

else:
    # Local machine: relative to this file
    PROJECT_ROOT = Path(__file__).resolve()
    for _ in range(3):  # try to go up until finding "datalake"
        if (PROJECT_ROOT / "datalake").exists():
            break
        PROJECT_ROOT = PROJECT_ROOT.parent

    DATA_DIR = PROJECT_ROOT / "datalake" / "curated"
    AI_LOCAL = DATA_DIR / "llm" / "ai_generated.csv"
    HUMAN_LOCAL = DATA_DIR / "twitter" / "high_quality_human.csv"

print(f"[path] AI_LOCAL={AI_LOCAL}")
print(f"[path] HUMAN_LOCAL={HUMAN_LOCAL}")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==== training params ====
EPOCHS = 4
EARLY_STOP = 2
F1_THRESHOLD = 0.73
PREC_MIN = 0.70         # avoid high-recall/low-precision flips
STABLE_EVALS = 2        # require N consecutive evals meeting thresholds
LR = 5e-6
SAMPLES = 15000
GRAD_ACCUM = 4          # update every 4 steps


LORA_CFG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    bias="none", task_type="SEQ_CLS",
    modules_to_save=["classifier"],   # ✅ let classifier not be frozen by PEFT
)



@dataclass
class RLConfig:
    group_size: int = 16
    epochs: int = 16
    train_batch_groups: int = 64
    eval_every_groups: int = 16
    lr: float = 5e-6
    max_len: int = 256
    kl_coef: float = 0.01
    bonus_confidence: float = 0.1
    seed: int = 42
    sample_per_class: int = 20000
    early_stop: int = 8  # stop if F1 not improving
    grad_accum: int = 1
RLCFG = RLConfig()


def _primary_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _to_primary_device(batch):
    dev = _primary_device()
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(dev, non_blocking=True)
    return batch


# ======================== MODEL ========================
class QwenJudge(nn.Module):
    def __init__(self, base):
        super().__init__()
        # model parallelism
        self.base = AutoModel.from_pretrained(
            base, token=HF_TOKEN,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.base.gradient_checkpointing_enable()
        self.base.config.use_cache = False

        self.config = self.base.config
        hidden = self.base.config.hidden_size

        # default on cuda:0
        self.classifier = nn.Linear(hidden, 2).to(torch.device("cuda:0"))

        try:
            self.base = prepare_model_for_kbit_training(self.base)
        except Exception as e:
            print(f"[warn] prepare_model_for_kbit_training skipped: {e}")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, 0, :]
        # ✅ move classifier to output device (pooled may not be on cuda:0)
        if next(self.classifier.parameters()).device != pooled.device:
            self.classifier.to(pooled.device)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], device=logits.device))
            loss = criterion(logits, labels)

        return {"loss": loss, "logits": logits}


class QwenEncoderJudge(nn.Module):
    def __init__(self, base_model_name: str, hf_token: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name, token=hf_token,
            device_map="auto",
            dtype=torch.bfloat16, trust_remote_code=True
        )
        self.base.gradient_checkpointing_enable()
        self.base.config.use_cache = False
        self.config = self.base.config
        hidden = self.base.config.hidden_size
        self.classifier = nn.Linear(hidden, 2).to(torch.device("cuda:0"))
        try:
            self.base = prepare_model_for_kbit_training(self.base)
        except Exception as e:
            print(f"[warn] prepare_model_for_kbit_training skipped: {e}")
            


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, 0, :]
        if next(self.classifier.parameters()).device != pooled.device:
            self.classifier.to(pooled.device)
        logits = self.classifier(pooled)
        return logits


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def collate(tokenizer, max_len):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        enc["labels"] = labels
        return enc
    return _fn

def compute_metrics(accelerator, model, tokenizer, loader):
    model.eval()
    y_true_parts, y_pred_parts = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ("input_ids","attention_mask","labels"):
                batch[k] = batch[k].to(accelerator.device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.55).long()

            y_pred_parts.append(preds)
            y_true_parts.append(batch["labels"])
    y_pred = accelerator.gather_for_metrics(torch.cat(y_pred_parts))
    y_true = accelerator.gather_for_metrics(torch.cat(y_true_parts))
    y_pred = y_pred.cpu().tolist(); y_true = y_true.cpu().tolist()
    acc = accuracy_score(y_true,y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true,y_pred,average="binary")
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}, y_true, y_pred

def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["Human","AI"], yticklabels=["Human","AI"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def get_latest_checkpoint(root="/mnt/cache"):
    """
    Find the most recent checkpoint directory (SFT or GRPO) under /mnt/cache.
    Priority: latest by mtime; will return even if 'best/' subdir is missing.
    """
    if not os.path.exists(root):
        print(f"[warn] Cache directory {root} not found.")
        return None

    candidates = []
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if os.path.isdir(full) and (d.startswith("sft_run_") or d.startswith("grpo_run_")):
            mtime = os.path.getmtime(full)
            candidates.append((mtime, full))

    if not candidates:
        print("❌ No SFT or GRPO checkpoints found under", root)
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dir = candidates[0][1]

    best_dir = os.path.join(latest_dir, "best")
    if os.path.isdir(best_dir):
        print(f"[auto] Using best checkpoint: {best_dir}")
        return best_dir
    else:
        print(f"[auto] Using latest checkpoint directory: {latest_dir}")
        return latest_dir


@app.function(image=image, volumes={"/mnt/cache": volume})
def resolve_latest_checkpoint():
    path = get_latest_checkpoint("/mnt/cache")  # Use the fixed version
    return path

# ======================== GRPO ========================
@app.function(image=image, gpu="A100-40GB:4", timeout=7200, volumes={"/mnt/cache": volume})
def train_grpo(ai_bytes: bytes, human_bytes: bytes, ref_dir: str = None):
    from tqdm import tqdm
    from sklearn.metrics import f1_score

    torch.set_float32_matmul_precision("high")

    # --- helpers ---
    def _primary_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _to_primary_device(batch):
        dev = _primary_device()
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(dev, non_blocking=True)
        return batch

    def evaluate_single_process(model, tokenizer, loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                batch = _to_primary_device(batch)
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                preds = logits.argmax(-1)
                y_pred += preds.cpu().tolist()
                y_true += batch["labels"].cpu().tolist()
        acc = accuracy_score(y_true, y_pred)
        pr, re, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        return {"accuracy": acc, "precision": pr, "recall": re, "f1": f1}, y_true, y_pred

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(CACHE_ROOT, f"grpo_run_{ts}")
    os.makedirs(RUN_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(RUN_DIR, "train_log.txt"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ---- data ----
    set_seed(RLCFG.seed)
    ai_df = pd.read_csv(io.BytesIO(ai_bytes)).sample(n=RLCFG.sample_per_class, random_state=RLCFG.seed)
    human_df = pd.read_csv(io.BytesIO(human_bytes)).sample(n=RLCFG.sample_per_class, random_state=RLCFG.seed)
    ai_df["label"] = 1
    human_df["label"] = 0
    df = pd.concat([ai_df, human_df]).sample(frac=1, random_state=RLCFG.seed)
    dataset = Dataset.from_pandas(df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    split = dataset.train_test_split(test_size=0.1, seed=RLCFG.seed)
    tr, ev = split["train"], split["test"]

    eval_loader = DataLoader(
        ev, batch_size=16, shuffle=False,
        collate_fn=collate(tokenizer, RLCFG.max_len),
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    full_loader = DataLoader(
        tr, batch_size=RLCFG.group_size, shuffle=True,
        collate_fn=collate(tokenizer, RLCFG.max_len),
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    # ---- models (policy & ref) ----
    # === 🧭 Auto-locate latest checkpoint if ref_dir missing ===
    def find_latest_checkpoint(root="/mnt/cache"):
        """Find latest GRPO or SFT run under /mnt/cache"""
        import re
        from glob import glob

        if not os.path.exists(root):
            print(f"[warn] Cache root {root} not found.")
            return None

        pattern = re.compile(r"(grpo_run|sft_run)_(\d{8}_\d{6})")
        candidates = []
        for d in glob(os.path.join(root, "*")):
            if os.path.isdir(d) and pattern.search(os.path.basename(d)):
                mtime = os.path.getmtime(d)
                candidates.append((mtime, d))

        if not candidates:
            print("❌ No previous GRPO/SFT checkpoints found.")
            return None

        latest_dir = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        best_dir = os.path.join(latest_dir, "best")
        if os.path.isdir(best_dir):
            print(f"[auto] Using latest checkpoint: {best_dir}")
            return best_dir
        else:
            print(f"[warn] No 'best' subfolder in {latest_dir}")
            return latest_dir

    if not ref_dir or not os.path.exists(ref_dir):
        print("[auto] --ref_best_dir not specified or not found, searching latest GRPO/SFT run...")
        ref_dir = find_latest_checkpoint("/mnt/cache")
        if not ref_dir:
            raise RuntimeError("❌ No checkpoint found. Please run SFT or GRPO first.")
        else:
            print(f"[auto] Auto-loaded latest checkpoint from {ref_dir}")

    # === 1️⃣ Load policy model ===
    policy_backbone = QwenEncoderJudge(BASE_MODEL, HF_TOKEN)  # model-parallel inside
    model = get_peft_model(policy_backbone, LORA_CFG)
    print("Attention impl:", getattr(model.base.config, "attn_impl", "unknown"))

    # === 2️⃣ Try loading SFT LoRA adapters for policy ===
    sft_adapters = os.path.join(ref_dir, "adapters") if ref_dir else None
    if sft_adapters and os.path.isdir(sft_adapters):
        bin_path = os.path.join(sft_adapters, "adapter_model.bin")
        safe_path = os.path.join(sft_adapters, "adapter_model.safetensors")
        ckpt_path = bin_path if os.path.exists(bin_path) else safe_path

        if os.path.exists(ckpt_path):
            print(f"[init] Loading SFT adapters into policy from {sft_adapters}")
            model.load_adapter(sft_adapters, adapter_name="sft")
            model.set_adapter("sft")
        else:
            print(f"[warn] No adapter weight file found in {sft_adapters}")
    else:
        print(f"[warn] Adapter directory not found under {ref_dir}")

    # === 3️⃣ Determine reference model path ===
    ref_name = None
    if ref_dir and os.path.isdir(ref_dir):
        backbone_path = os.path.join(ref_dir, "backbone")
        if os.path.isdir(backbone_path):
            ref_name = backbone_path
            print(f"[ref] Using backbone snapshot from {backbone_path}")
        elif os.path.exists(os.path.join(ref_dir, "config.json")):
            ref_name = ref_dir
            print(f"[ref] Using {ref_dir} as reference model")
        else:
            print(f"[warn] {ref_dir} has no config.json, fallback to BASE_MODEL.")
            ref_name = BASE_MODEL
    else:
        maybe_latest = get_latest_checkpoint("/mnt/cache")
        if maybe_latest and os.path.isdir(os.path.join(maybe_latest, "backbone")):
            ref_name = os.path.join(maybe_latest, "backbone")
            print(f"[ref] Using latest checkpoint backbone from {ref_name}")
        else:
            print(f"[warn] No SFT checkpoint found; using BASE_MODEL as reference")
            ref_name = BASE_MODEL

    # === 4️⃣ Load reference model ===
    ref_base = QwenEncoderJudge(ref_name, HF_TOKEN)
    print(f"[init] Reference model loaded from {ref_name}")

    # === 5️⃣ Load same SFT LoRA adapters into reference model (frozen) ===
    if sft_adapters and os.path.isdir(sft_adapters):
        bin_path = os.path.join(sft_adapters, "adapter_model.bin")
        safe_path = os.path.join(sft_adapters, "adapter_model.safetensors")
        ckpt_path = bin_path if os.path.exists(bin_path) else safe_path

        if os.path.exists(ckpt_path):
            print(f"[init] Loading same SFT adapters into reference from {sft_adapters}")
            ref_base = get_peft_model(ref_base, LORA_CFG)
            ref_base.load_adapter(sft_adapters, adapter_name="sft_ref")
            ref_base.set_adapter("sft_ref")
        else:
            print(f"[warn] No adapter weights found for reference in {sft_adapters}")
    else:
        print(f"[warn] Adapter directory not found for reference model under {ref_dir}")

    # === 6️⃣ Freeze reference model ===
    for p in ref_base.parameters():
        p.requires_grad_(False)
    ref_base.eval()

    print("==== [GRPO INIT] Models ready: policy (trainable) + reference (frozen) ====")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=RLCFG.lr)

    writer = SummaryWriter(log_dir=os.path.join(RUN_DIR, "tb"))
    try:
        wandb.finish()
    except Exception:
        pass
    wandb.init(
        project="TweetVerify",
        name=f"GRPO_{ts}",
        reinit=True,
        id=str(uuid.uuid4()),
        resume="never"
    )

    global_group, best_f1, patience = 0, -1.0, 0
    best_dir = None

    for epoch in range(1, RLCFG.epochs + 1):
        model.train()
        pbar = tqdm(full_loader, total=RLCFG.train_batch_groups, desc=f"[GRPO] epoch {epoch}")

        for groups_done, batch in enumerate(pbar, start=1):
            batch = _to_primary_device(batch)

            # ============================================================
            # 0. Forward pass: logits → probs → actions
            # ============================================================
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # If your QwenEncoderJudge returns {"logits": ...}
            if isinstance(logits, dict):
                logits = logits["logits"]

            probs = torch.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)

            # ============================================================
            # 1. Compute extended verifiable reward (no PPO)
            # ============================================================

            idx = torch.arange(actions.size(0), device=actions.device)
            logp = torch.log_softmax(logits, dim=-1)

            # correctness reward (primary RLVR signal)
            correct = (actions == batch["labels"]).float()

            # logit margin: log p(true) - log p(false)
            logp_true = logp[idx, batch["labels"]]
            logp_false = logp[idx, 1 - batch["labels"]]
            margin = logp_true - logp_false

            # batch-level F1 shaping (optional but allowed)
            y_true = batch["labels"].detach().cpu().numpy()
            y_pred = actions.detach().cpu().numpy()
            try:
                f1_batch = f1_score(y_true, y_pred)
            except:
                f1_batch = 0.5

            # Final shaped reward
            reward = (
                1.0 * correct +      # verifiable correctness
                0.3 * margin +       # decision sharpness
                0.4 * (f1_batch - 0.5)
            )

            # ============================================================
            # 2. GRPO group-relative advantage A_k = r_k - mean(r)
            # ============================================================
            group_mean = reward.mean().detach()
            advantage = reward - group_mean

            # ============================================================
            # 3. Policy gradient (REINFORCE) — NO PPO, NO clipping
            # ============================================================
            chosen_logp = logp[idx, actions]
            pg_loss = -(advantage * chosen_logp).mean()

            # ============================================================
            # 4. KL(π || π_ref) penalty with frozen reference model
            # ============================================================
            with torch.no_grad():
                ref_logits = ref_base(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                if isinstance(ref_logits, dict):
                    ref_logits = ref_logits["logits"]

                ref_logp = torch.log_softmax(ref_logits, dim=-1)

            kl = (probs * (logp - ref_logp)).sum(dim=-1).mean()

            # ============================================================
            # 5. Final GRPO loss
            # ============================================================
            loss = pg_loss + RLCFG.kl_coef * kl

            # backward + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            global_group += 1

            if global_group % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_group)
                wandb.log({"loss": loss.item(), "r_mean": group_mean.item(), "kl": kl.item(), "step": global_group})

            if groups_done >= RLCFG.train_batch_groups:
                break

            # eval periodically
            if global_group % RLCFG.eval_every_groups == 0:
                metrics, y_true_e, y_pred_e = evaluate_single_process(model, tokenizer, eval_loader)
                wandb.log(metrics)
                writer.add_scalar("val/f1", metrics["f1"], global_group)
                logging.info(f"[eval] {metrics}")

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    patience = 0
                    best_dir = os.path.join(RUN_DIR, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    # save snapshot separately
                    tokenizer.save_pretrained(best_dir)
                    model.save_pretrained(os.path.join(best_dir, "adapters"))
                    backbone = getattr(getattr(model, "base", None), "base", None)
                    if backbone is not None:
                        backbone.save_pretrained(os.path.join(best_dir, "backbone"))
                    # confusion matrix
                    plot_confusion(y_true_e, y_pred_e, os.path.join(best_dir, "confusion_matrix.png"))
                    with open(os.path.join(best_dir, "metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)
                    volume.commit()

                else:
                    if epoch <= 2:
                        logging.info(f"[warmup] Epoch {epoch}: no F1 improvement, but patience not counted yet.")
                        continue  # skip early stop logic

                    patience += 1
                    logging.info(f"[train] No F1 improvement, patience = {patience}/{RLCFG.early_stop}")

                    if patience >= RLCFG.early_stop:
                        logging.info("Early stopping (no F1 improvement after warmup).")
                        writer.close()
                        wandb.finish()
                        return {"run_dir": RUN_DIR, "best_dir": best_dir, "best_f1": best_f1}


        # epoch-end snapshot
        epoch_dir = os.path.join(RUN_DIR, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        tokenizer.save_pretrained(epoch_dir)
        model.save_pretrained(os.path.join(epoch_dir, "adapters"))
        backbone = getattr(getattr(model, "base", None), "base", None)
        if backbone is not None:
            backbone.save_pretrained(os.path.join(epoch_dir, "backbone"))
        volume.commit()

    writer.close()
    wandb.finish()
    return {"run_dir": RUN_DIR, "best_dir": best_dir, "best_f1": best_f1}



# ======================== SFT LORA ========================
@app.function(image=image, gpu="A100-40GB:4", volumes={"/mnt/cache": volume}, timeout=7200)
def train_sft(ai_bytes: bytes, human_bytes: bytes, ref_best_dir: str = None):
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    import wandb

    torch.set_float32_matmul_precision("high")

    # --- small helpers ---
    def _primary_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _to_primary_device(batch):
        dev = _primary_device()
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(dev, non_blocking=True)
        return batch

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = f"/mnt/cache/sft_run_{ts}"
    os.makedirs(RUN_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(RUN_DIR, "train.log"), level=logging.INFO)

    # ------------------------------------------------------------
    # WARM START (tokenizer + base + LoRA)
    # ------------------------------------------------------------
    print("===== SFT Warm-Start =====")
    print(f"[SFT] ref_dir = {ref_best_dir}")

    # 1. tokenizer
    if ref_best_dir and os.path.exists(ref_best_dir):
        try:
            print(f"[SFT] Warm-start tokenizer from {ref_best_dir}")
            tokenizer = AutoTokenizer.from_pretrained(
                ref_best_dir, trust_remote_code=True, token=HF_TOKEN
            )
        except Exception as e:
            print(f"[SFT] Failed to load tokenizer: {e}")
            print("[SFT] → Falling back to BASE_MODEL tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL, trust_remote_code=True, token=HF_TOKEN
            )
    else:
        print("[SFT] No previous tokenizer found — BASE_MODEL tokenizer used")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=HF_TOKEN
        )

    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. backbone (always base)
    print("[SFT] Loading BASE_MODEL backbone")
    base = QwenJudge(BASE_MODEL)

    # 3. LoRA wrapping
    model = get_peft_model(base, LORA_CFG)

    # 4. attempt to load LoRA adapters
    adapter_dir = (
        os.path.join(ref_best_dir, "adapters")
        if ref_best_dir else None
    )

    if adapter_dir and os.path.exists(adapter_dir):
        try:
            print(f"[SFT] Warm-starting LoRA adapters from {adapter_dir}")
            model.load_adapter(adapter_dir, adapter_name="default")
            model.set_adapter("default")
        except Exception as e:
            print(f"[SFT] Failed to load LoRA adapter: {e}")
            print("[SFT] → Training LoRA from scratch.")
    else:
        print("[SFT] No LoRA adapters found — training LoRA from scratch.")

    print("===== SFT Warm-Start Complete =====")
    print("Attention impl:", getattr(model.base.config, "attn_impl", "unknown"))

    # ------------------------------------------------------------
    # Dataset loading (AFTER tokenizer exists)
    # ------------------------------------------------------------
    ai_df = pd.read_csv(io.BytesIO(ai_bytes)).sample(n=SAMPLES, random_state=42)
    human_df = pd.read_csv(io.BytesIO(human_bytes)).sample(n=SAMPLES, random_state=42)
    ai_df["label"] = 1
    human_df["label"] = 0
    df = pd.concat([ai_df, human_df]).sample(frac=1, random_state=42)
    dataset = Dataset.from_pandas(df[["text", "label"]])

    split = dataset.train_test_split(test_size=0.1, seed=42)
    tr, ev = split["train"], split["test"]

    def make_loader(ds, bsz):
        return DataLoader(
            ds,
            batch_size=bsz,
            shuffle=True,
            collate_fn=collate(tokenizer, 256),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    train_loader = make_loader(tr, 8)
    eval_loader = make_loader(ev, 16)

    # ------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    sch = get_linear_schedule_with_warmup(
        opt, 0, int(len(train_loader) * EPOCHS / GRAD_ACCUM) + 1
    )

    writer = SummaryWriter(log_dir=os.path.join(RUN_DIR, "tb"))
    try:
        wandb.finish()
    except:
        pass

    wandb.init(
        project="TweetVerify",
        name=f"SFT_{ts}",
        reinit=True,
        id=str(uuid.uuid4()),
        resume="never"
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def evaluate():
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for b in eval_loader:
                b = _to_primary_device(b)
                out = model(**b)
                preds = out["logits"].argmax(-1)
                y_pred += preds.cpu().tolist()
                y_true += b["labels"].cpu().tolist()
        acc = accuracy_score(y_true, y_pred)
        pr, re, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        return {"accuracy": acc, "precision": pr, "recall": re, "f1": f1}

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_f1, patience, stable_hits = -1.0, 0, 0
    global_step = 0

    model.train()
    for ep in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"SFT {ep}")
        opt.zero_grad(set_to_none=True)

        for step, b in enumerate(pbar, start=1):
            b = _to_primary_device(b)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(**b)
                loss = out["loss"] / GRAD_ACCUM

            loss.backward()

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                wandb.log({
                    "train/loss": float(loss.item() * GRAD_ACCUM),
                    "epoch": ep,
                    "step": global_step
                })

        # Eval
        m = evaluate()
        writer.add_scalar("val/f1", m["f1"], ep)
        writer.add_scalar("val/precision", m["precision"], ep)
        writer.add_scalar("val/recall", m["recall"], ep)
        wandb.log(m)
        logging.info(f"Epoch {ep} {m}")

        # F1 early stop
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOP:
            logging.info("Early stop (no F1 improvement).")
            break

        # Auto trigger GRPO
        if (m["f1"] >= F1_THRESHOLD) and (m["precision"] >= PREC_MIN):
            stable_hits += 1
        else:
            stable_hits = 0

        if stable_hits >= STABLE_EVALS:
            logging.info("Triggering GRPO...")
            train_grpo.spawn(ai_bytes, human_bytes, f"/mnt/cache/sft_run_{ts}/best/backbone")
            break

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------
    save_dir = f"/mnt/cache/sft_run_{ts}/best"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(os.path.join(save_dir, "adapters"))

    backbone = getattr(getattr(model, "base", None), "base", None)
    if backbone is not None:
        backbone.save_pretrained(os.path.join(save_dir, "backbone"))

    volume.commit()

    return {"best_f1": best_f1, "run_dir": save_dir}



# ======================== CHECKPOINT UTILITIES ========================
@app.function(image=image, volumes={"/mnt/cache": volume})
def list_checkpoints():
    files = os.listdir(CACHE_ROOT)
    runs = [f for f in files if f.startswith(("sft_run","grpo_run","active_"))]
    return sorted(runs)

@app.function(image=image, volumes={"/mnt/cache": volume})
def rollback_checkpoint(checkpoint_name: str):
    src = os.path.join(CACHE_ROOT, checkpoint_name)
    if not os.path.exists(src):
        return {"error": f"Checkpoint {checkpoint_name} not found"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(CACHE_ROOT, f"active_{ts}")
    os.system(f"cp -r {src} {dst}")

    symlink_path = os.path.join(CACHE_ROOT, "latest_active")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(dst, symlink_path)

    logging.info(f"Rolled back to {dst}, symlink updated.")
    volume.commit()
    return {"rolled_back_to": dst, "symlink": symlink_path}

# ======================== ENTRYPOINT ========================
# ======================== ENTRYPOINT ========================
@app.local_entrypoint()
def main(
    cmd: str = "sft",
    ref_best_dir: str = None,
    checkpoint_name: str = None,
):
    if cmd == "sft":
        # Allow resume or warm-start from latest checkpoint
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if ref_dir:
            print(f"ℹ️  (optional) Using existing checkpoint for SFT initialization: {ref_dir}")
        else:
            print("⚠️  No previous checkpoint found — training will start from BASE_MODEL.")

        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            res = train_sft.remote(f1.read(), f2.read())
            print("✅ SFT job finished:", res)

    elif cmd == "grpo":
        ref_dir = ref_best_dir or resolve_latest_checkpoint.remote()

        if not ref_dir:
            print("❌ No checkpoint found in /mnt/cache. Please run SFT first or specify --ref_best_dir.")
            return

        print(f"Using reference checkpoint: {ref_dir}")
        with open(AI_LOCAL, "rb") as f1, open(HUMAN_LOCAL, "rb") as f2:
            res = train_grpo.remote(f1.read(), f2.read(), ref_dir)
            print("✅ GRPO job finished:", res)