import torch
from torch.utils.data import DataLoader
from src.utils.collate_batch import collate_batch


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        num_workers=1
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.num_workers = num_workers

    @torch.no_grad()
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
        if self.model.get_name() == 'bert':
            dataloader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True)
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                z = self.model(input_ids, attention_mask)
                y = torch.argmax(z, dim=1)
                correct += (y == labels).sum().item()
                total += labels.size(0)
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                collate_fn=collate_batch,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            for i, (x, t) in enumerate(dataloader):
                x, t = x.to(self.device), t.to(self.device)
                z = self.model(x)
                y = torch.argmax(z, dim=1)
                correct += (y == t).sum().item()
                total += t.size(0)

        acc = correct / total

        return acc
