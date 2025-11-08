"""
Main training script for RLVR pipeline with hyperparameter ablations.
"""
import argparse
import os
from pathlib import Path
import json
try:
    from .data_loader import load_data, create_splits
    from .rlvr_trainer import RLVRTrainer
except ImportError:
    from data_loader import load_data, create_splits
    from rlvr_trainer import RLVRTrainer


def main():
    parser = argparse.ArgumentParser(description="RLVR Training Pipeline")
    
    # Data arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per class")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio")
    
    # Model arguments
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased",
                       help="Base model to fine-tune")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of epochs")
    
    # RL arguments
    parser.add_argument("--reward-metric", type=str, default="f1",
                       choices=["f1", "accuracy"],
                       help="Metric for reward computation")
    parser.add_argument("--kl-penalty", type=float, default=0.1,
                       help="KL divergence penalty coefficient")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                       help="PPO clipping epsilon")
    parser.add_argument("--value-coef", type=float, default=0.5,
                       help="Value function loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                       help="Entropy bonus coefficient")
    
    # Logging arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="rlvr-tweet-verification",
                       help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="wandb run name")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./models",
                       help="Output directory for models")
    parser.add_argument("--save-model", action="store_true",
                       help="Save trained model")
    
    # Experiment config (for ablations)
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file for hyperparameters")
    
    args = parser.parse_args()
    
    # Load hyperparameters from config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    df = load_data(max_samples=args.max_samples)
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Initialize trainer
    print("\n" + "="*60)
    print("Initializing RLVR Trainer")
    print("="*60)
    print(f"Base Model: {args.base_model}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Reward Metric: {args.reward_metric}")
    print(f"KL Penalty: {args.kl_penalty}")
    print(f"Clip Epsilon: {args.clip_epsilon}")
    print(f"Value Coef: {args.value_coef}")
    print(f"Entropy Coef: {args.entropy_coef}")
    
    trainer = RLVRTrainer(
        base_model_name=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        reward_metric=args.reward_metric,
        kl_penalty=args.kl_penalty,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project
    )
    
    # Set wandb run name if provided
    if args.use_wandb and args.wandb_run_name:
        import wandb
        wandb.run.name = args.wandb_run_name
    
    # Train
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    trainer.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        val_texts=val_df['text'].tolist(),
        val_labels=val_df['label'].tolist()
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    test_metrics = trainer.evaluate(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist()
    )
    
    print("\nFinal Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Log test metrics to wandb
    if args.use_wandb:
        import wandb
        for metric, value in test_metrics.items():
            wandb.log({f"test/{metric}": value})
    
    # Save model
    if args.save_model:
        model_path = output_dir / f"rlvr_{args.base_model.replace('/', '_')}"
        trainer.save_model(str(model_path))
        
        # Save training config
        config_path = model_path / "training_config.json"
        config_dict = {
            "base_model": args.base_model,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "reward_metric": args.reward_metric,
            "kl_penalty": args.kl_penalty,
            "clip_epsilon": args.clip_epsilon,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "test_metrics": test_metrics
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"\nTraining config saved to {config_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

