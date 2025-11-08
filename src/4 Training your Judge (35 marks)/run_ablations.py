"""
Script to run hyperparameter ablation studies.
"""
import json
import subprocess
import os
from pathlib import Path


# Hyperparameter configurations for ablations
ABLATION_CONFIGS = [
    # Baseline configuration
    {
        "name": "baseline",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # Learning rate ablations
    {
        "name": "lr_1e-5",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 1e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    {
        "name": "lr_5e-5",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # Batch size ablations
    {
        "name": "batch_8",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    {
        "name": "batch_32",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # KL penalty ablations
    {
        "name": "kl_0.05",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.05,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    {
        "name": "kl_0.2",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.2,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # Clip epsilon ablations
    {
        "name": "clip_0.1",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.1,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    {
        "name": "clip_0.3",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.3,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # Reward metric ablation
    {
        "name": "reward_accuracy",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "accuracy",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01
    },
    # Entropy coefficient ablations
    {
        "name": "entropy_0.0",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.0
    },
    {
        "name": "entropy_0.05",
        "base_model": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "reward_metric": "f1",
        "kl_penalty": 0.1,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.05
    },
]


def run_ablation(config: dict, output_dir: str = "./ablation_results"):
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running ablation: {config['name']}")
    print(f"{'='*60}")
    
    # Create config file
    config_dir = Path(output_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config['name']}.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build command
    cmd = [
        "python", "train.py",
        "--config", str(config_path),
        "--use-wandb",
        "--wandb-run-name", f"ablation_{config['name']}",
        "--output-dir", str(Path(output_dir) / "models" / config['name']),
        "--save-model"
    ]
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Ablation {config['name']} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ablation {config['name']} failed:")
        print(e.stderr)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config-idx", type=int, default=None,
                       help="Run specific ablation by index (0-based)")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples per class for faster ablations")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save all configs
    all_configs_path = Path(args.output_dir) / "all_configs.json"
    with open(all_configs_path, 'w') as f:
        json.dump(ABLATION_CONFIGS, f, indent=2)
    print(f"Saved {len(ABLATION_CONFIGS)} ablation configurations to {all_configs_path}")
    
    # Run ablations
    if args.config_idx is not None:
        # Run single ablation
        config = ABLATION_CONFIGS[args.config_idx]
        run_ablation(config, args.output_dir)
    else:
        # Run all ablations
        print(f"\nRunning {len(ABLATION_CONFIGS)} ablation experiments...")
        results = []
        for i, config in enumerate(ABLATION_CONFIGS):
            success = run_ablation(config, args.output_dir)
            results.append({
                "name": config['name'],
                "success": success
            })
        
        # Summary
        print(f"\n{'='*60}")
        print("Ablation Summary")
        print(f"{'='*60}")
        successful = sum(1 for r in results if r['success'])
        print(f"Successful: {successful}/{len(results)}")
        print(f"\nResults:")
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"  {status} {r['name']}")


if __name__ == "__main__":
    main()


