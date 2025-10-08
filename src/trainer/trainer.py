import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.collate_batch import collate_batch
from src.evaluator.evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        device,
        model,
        train_data,
        val_data,
        learning_rate=1e-4,
        batch_size=314,
        num_epochs=10,num_workers=1
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_evaluator = Evaluator(self.model, self.train_data, self.device)
        self.val_evaluator = Evaluator(self.model, self.val_data, self.device)
        self.num_workers=num_workers

    def train_model(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_batch,
            shuffle=True,num_workers=self.num_workers
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        train_loss, train_acc, val_acc = [], [], []
        best_val_acc = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            ta = self.train_evaluator.accuracy(self.batch_size)
            va = self.val_evaluator.accuracy(self.batch_size)

            if va > best_val_acc:
                best_val_acc = va
                torch.save(self.model.state_dict(), "best_model.pth")

            train_loss.append(avg_loss)
            train_acc.append(ta)
            val_acc.append(va)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {ta:.2f}% | Val Acc: {va:.2f}%"
            )

        print("Training complete. Best Val Acc:", best_val_acc)
        return train_loss, train_acc, val_acc
