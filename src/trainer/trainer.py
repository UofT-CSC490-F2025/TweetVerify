import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.collate_batch import collate_batch
from src.evaluator.evaluator import Evaluator
import os
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        learning_rate: float = 1e-4,
        batch_size: int = 314,
        num_epochs: int = 10,
        num_workers: int = 1,
        model_save_dir: str | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers

        self.train_evaluator = Evaluator(
            self.model, self.train_data, self.device, num_workers
        )
        self.val_evaluator = Evaluator(
            self.model, self.val_data, self.device, num_workers
        )

        if model_save_dir is None:
            self.model_save_dir = os.getenv("SM_MODEL_DIR", "./models")
        else:
            self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

        model_name = self.model.get_name()

        # DataLoader
        if model_name in ["rnn", "lstm"]:
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                collate_fn=collate_batch,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def train_model(self):

        criterion = nn.CrossEntropyLoss()

        model_name = self.model.get_name()
        if model_name in ["lstm", "rnn"]:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            use_step_scheduler_per_batch = False
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
            )
            total_steps = len(self.train_loader) * self.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )
            use_step_scheduler_per_batch = True

        train_loss = []
        val_auc = []

        best_val_auc = -1.0
        best_model_path = ""

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            if model_name in ["lstm", "rnn"]:
                for texts, labels in self.train_loader:
                    texts = texts.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                scheduler.step()

            else:

                for batch in tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                    ncols=80,
                ):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    optimizer.zero_grad()

                    if model_name == "roberta_extra":
                        extra_features = batch["extra_features"].to(self.device)
                        outputs = self.model(input_ids, attention_mask, extra_features)
                    else:
                        outputs = self.model(input_ids, attention_mask)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=3.0
                    )
                    optimizer.step()
                    if use_step_scheduler_per_batch:
                        scheduler.step()

                    epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            acc, f1, auc = self.val_evaluator.accuracy(self.batch_size)
            if auc > best_val_auc:
                best_val_auc = auc
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_path = os.path.join(
                    self.model_save_dir,
                    f"{model_name}_{round(auc * 100, 1)}_{timestamp}.pt",
                )
                torch.save(self.model.state_dict(), model_path)
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = model_path

            train_loss.append(avg_loss)
            val_auc.append(auc)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Val AUC: {auc*100:.2f}% | "
                f"Val Acc: {acc*100:.2f}% | "
                f"Val F1: {f1*100:.2f}%",
                flush=True,
            )

        print(f"Training complete. Best Val AUC: {best_val_auc:.4f}", flush=True)
        if best_model_path:
            state_dict = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded best model from {best_model_path}", flush=True)

        return train_loss, val_auc
