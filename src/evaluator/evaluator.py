import torch
from torch.utils.data import DataLoader
from src.utils.collate_batch import collate_batch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        num_workers=1,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.num_workers = num_workers

    def accuracy(self, batch_size: int = 64):
        """
        Estimate the accuracy of the model over the dataset.
        The predicted class is the one with the highest probability (argmax).

        Parameters:
            batch_size (int): Batch size for DataLoader.

        Returns:
            float: Accuracy between 0 and 1.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        if self.model.get_name() in ["lstm", "rnn"]:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                collate_fn=collate_batch,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,  
                num_workers=self.num_workers,
            )

        with torch.no_grad():
            for batch in dataloader:
                if self.model.get_name() in ["lstm", "rnn"]:
                    x, t = batch
                    x, t = x.to(self.device), t.to(self.device)
                    z = self.model(x)
                    labels = t
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    if self.model.get_name() == "roberta_extra":
                        extra_features = batch["extra_features"].to(self.device)
                        z = self.model(input_ids, attention_mask, extra_features)
                    else:
                        z = self.model(input_ids, attention_mask)
                y_pred = torch.argmax(z, dim=1)
                all_preds.append(y_pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.softmax(z, dim=1).cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs=np.concatenate(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)   
        auc = roc_auc_score(all_labels, all_probs[:, 1]) 

        return acc, f1, auc
