import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.collate_batch import collate_batch
from src.evaluator.evaluator import Evaluator
import os
from datetime import datetime
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        device,
        model,
        train_data,
        val_data,
        learning_rate=1e-4,
        batch_size=314,
        num_epochs=10, num_workers=1,
        model_save_dir=None
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_evaluator = Evaluator(
            self.model, self.train_data, self.device)
        self.val_evaluator = Evaluator(self.model, self.val_data, self.device)
        self.num_workers = num_workers
        if not model_save_dir:
            self.model_save_dir = os.environ['SM_MODEL_DIR']
        else:
            self.model_save_dir = model_save_dir
        if self.model.get_name() == 'bert':
            self.train_loader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=batch_size,
                collate_fn=collate_batch,
                shuffle=True, num_workers=self.num_workers
            )

    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5)

        train_loss, train_acc, val_acc = [], [], []
        best_val_acc = 0
        best_model_path = ''
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            if self.model.get_name() == 'bert':
                for batch in self.train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                for texts, labels in self.train_loader:
                    texts, labels = texts.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(self.train_loader)
            ta = self.train_evaluator.accuracy(self.batch_size)
            va = self.val_evaluator.accuracy(self.batch_size)

            if va > best_val_acc:
                best_val_acc = va
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_path = os.path.join(
                    self.model_save_dir, f"{self.model.get_name()}_{round(va*100, 1)}_{timestamp}.pt")
                torch.save(self.model.state_dict(), model_path)
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = model_path

            train_loss.append(avg_loss)
            train_acc.append(ta)
            val_acc.append(va)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {ta*100:.2f}% | Val Acc: {va*100:.2f}%"
            , flush=True)

        print("Training complete. Best Val Acc:", best_val_acc, flush=True)
        return train_loss, train_acc, val_acc
