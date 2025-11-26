import torch
from torch.utils.data import DataLoader
from src.utils.collate_batch import collate_batch


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

    def accuracy(self, batch_size: int = 64) -> float:
        """
        Estimate the accuracy of the model over the dataset.
        The predicted class is the one with the highest probability (argmax).

        Parameters:
            batch_size (int): Batch size for DataLoader.

        Returns:
            float: Accuracy between 0 and 1.
        """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            if self.model.get_name() in ["lstm", "rnn"]:
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    collate_fn=collate_batch,
                    pin_memory=True,
                )

                for i, (x, t) in enumerate(dataloader):
                    x, t = x.to(self.device), t.to(self.device)
                    z = self.model(x)
                    y = torch.argmax(z, dim=1)
                    correct += (y == t).sum().item()
                    total += t.size(0)

            else:
                dataloader = DataLoader(
                    self.dataset, batch_size=batch_size, shuffle=True
                )
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    if self.model.get_name() == "roberta_extra":
                        extra_features = batch["extra_features"].to(self.device)
                        z = self.model(input_ids, attention_mask, extra_features)
                    else:
                        z = self.model(input_ids, attention_mask)
                    y = torch.argmax(z, dim=1)
                    correct += (y == labels).sum().item()
                    total += labels.size(0)
                    del input_ids, attention_mask, labels, z, y

        acc = correct / total

        return acc
