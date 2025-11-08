"""
RLVR (Reinforcement Learning with Verifier Reward) Training Pipeline.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import wandb
import os
try:
    from .model_wrapper import ClassificationModelWrapper
except ImportError:
    from model_wrapper import ClassificationModelWrapper
from sklearn.metrics import f1_score, accuracy_score


class TweetDataset(Dataset):
    """Dataset for tweet classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RLVRTrainer:
    """
    RLVR Trainer using PPO (Proximal Policy Optimization) for fine-tuning.
    """
    
    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        reward_metric: str = "f1",
        kl_penalty: float = 0.1,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
        use_wandb: bool = True,
        project_name: str = "rlvr-tweet-verification"
    ):
        """
        Initialize RLVR Trainer.
        
        Args:
            base_model_name: Base model to fine-tune
            learning_rate: Learning rate for optimization
            batch_size: Training batch size
            num_epochs: Number of training epochs
            reward_metric: Metric for reward computation ('f1', 'accuracy')
            kl_penalty: KL divergence penalty coefficient
            clip_epsilon: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device to use
            use_wandb: Whether to use wandb logging
            project_name: wandb project name
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reward_metric = reward_metric
        self.kl_penalty = kl_penalty
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model_wrapper = ClassificationModelWrapper(
            model_name=base_model_name,
            device=self.device
        )
        self.model = self.model_wrapper.get_model()
        self.tokenizer = self.model_wrapper.get_tokenizer()
        
        # Reference model for KL penalty (frozen copy)
        self.ref_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2
        ).to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "base_model": base_model_name,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "reward_metric": reward_metric,
                    "kl_penalty": kl_penalty,
                    "clip_epsilon": clip_epsilon,
                    "value_coef": value_coef,
                    "entropy_coef": entropy_coef
                }
            )
    
    def compute_reward(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """Compute reward based on selected metric."""
        if self.reward_metric == "f1":
            return float(f1_score(true_labels, predictions, average='binary', zero_division=0))
        elif self.reward_metric == "accuracy":
            return float(accuracy_score(true_labels, predictions))
        else:
            raise ValueError(f"Unknown reward metric: {self.reward_metric}")
    
    def compute_kl_divergence(
        self,
        logits_current: torch.Tensor,
        logits_ref: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference model."""
        probs_current = F.softmax(logits_current, dim=-1)
        probs_ref = F.softmax(logits_ref, dim=-1)
        
        kl = torch.sum(
            probs_current * torch.log(probs_current / (probs_ref + 1e-8) + 1e-8),
            dim=-1
        )
        return kl.mean()
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform one training step with PPO.
        
        Args:
            batch: Batch of tokenized inputs
            true_labels: Ground truth labels for reward computation
        
        Returns:
            Dictionary of loss components
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass through current model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Forward pass through reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
        
        # Get predictions for reward
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        reward = self.compute_reward(predictions, true_labels)
        
        # Compute policy (action probabilities)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get reference probabilities
        ref_probs = F.softmax(ref_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute importance sampling ratio
        # For each sample, use the predicted class probability
        action_indices = torch.argmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        selected_ref_log_probs = ref_log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        ratio = torch.exp(selected_log_probs - selected_ref_log_probs)
        
        # Compute advantages (reward - baseline)
        # Use mean reward as baseline
        advantages = torch.tensor(reward - 0.5, device=self.device).expand_as(ratio)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL penalty
        kl_div = self.compute_kl_divergence(logits, ref_logits)
        kl_penalty_loss = self.kl_penalty * kl_div
        
        # Entropy bonus
        entropy = self.compute_entropy(logits)
        entropy_bonus = -self.entropy_coef * entropy
        
        # Value function loss (simplified: use reward as target)
        value_target = torch.tensor(reward, device=self.device)
        # For simplicity, we'll use the mean logit as a value estimate
        value_estimate = logits.mean()
        value_loss = self.value_coef * F.mse_loss(value_estimate.unsqueeze(0), value_target.unsqueeze(0))
        
        # Total loss
        total_loss = policy_loss + kl_penalty_loss - entropy_bonus + value_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty_loss.item(),
            "entropy": entropy.item(),
            "entropy_bonus": entropy_bonus.item(),
            "value_loss": value_loss.item(),
            "reward": reward,
            "kl_div": kl_div.item()
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ):
        """
        Train the model using RLVR.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
        """
        # Create dataset and dataloader
        train_dataset = TweetDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        global_step = 0
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            epoch_losses = []
            epoch_rewards = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                # Get true labels for this batch
                batch_indices = range(
                    batch_idx * self.batch_size,
                    min((batch_idx + 1) * self.batch_size, len(train_texts))
                )
                batch_labels = np.array([train_labels[i] for i in batch_indices])
                
                # Training step
                loss_dict = self.train_step(batch, batch_labels)
                
                epoch_losses.append(loss_dict["total_loss"])
                epoch_rewards.append(loss_dict["reward"])
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "reward": f"{loss_dict['reward']:.4f}"
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss_dict["total_loss"],
                        "train/policy_loss": loss_dict["policy_loss"],
                        "train/kl_penalty": loss_dict["kl_penalty"],
                        "train/entropy": loss_dict["entropy"],
                        "train/value_loss": loss_dict["value_loss"],
                        "train/reward": loss_dict["reward"],
                        "train/kl_div": loss_dict["kl_div"],
                        "epoch": epoch,
                        "global_step": global_step
                    })
                
                global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")
            
            # Validation
            if val_texts is not None and val_labels is not None:
                val_metrics = self.evaluate(val_texts, val_labels)
                print(f"\nValidation Metrics:")
                for metric, value in val_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                if self.use_wandb:
                    for metric, value in val_metrics.items():
                        wandb.log({
                            f"val/{metric}": value,
                            "epoch": epoch
                        })
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, float]:
        """Evaluate model on given texts and labels."""
        predictions = self.model_wrapper.predict(texts)
        
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score
        )
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='binary', zero_division=0),
            "precision": precision_score(labels, predictions, average='binary', zero_division=0),
            "recall": recall_score(labels, predictions, average='binary', zero_division=0)
        }
    
    def save_model(self, path: str):
        """Save the trained model."""
        self.model_wrapper.save(path)
        print(f"[RLVR Trainer] Model saved to {path}")

