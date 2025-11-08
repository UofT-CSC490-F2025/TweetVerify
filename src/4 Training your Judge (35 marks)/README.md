# RLVR Training Pipeline: Training your Judge

This folder contains the implementation of a Reinforcement Learning with Verifier Reward (RLVR) training pipeline for fine-tuning a transformer-based classifier on the AI tweet detection task.

## Overview

The pipeline uses Proximal Policy Optimization (PPO) to fine-tune a DistilBERT model, optimizing for F1-score as the primary metric. The training directly optimizes the target metric through reward signals, allowing the model to learn better classification boundaries.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

Train the model with default hyperparameters:

```bash
python train.py \
    --base-model distilbert-base-uncased \
    --learning-rate 2e-5 \
    --batch-size 16 \
    --num-epochs 3 \
    --reward-metric f1 \
    --use-wandb \
    --save-model \
    --output-dir ./models
```

### Run Ablation Studies

Run all hyperparameter ablations:

```bash
python run_ablations.py --output-dir ./ablation_results
```

Run a specific ablation:

```bash
python run_ablations.py --config-idx 0 --output-dir ./ablation_results
```

## File Structure

- `model_wrapper.py`: Wrapper for transformer-based classification models
- `rlvr_trainer.py`: PPO-based RLVR trainer implementation
- `data_loader.py`: Data loading and splitting utilities
- `train.py`: Main training script
- `run_ablations.py`: Script for running hyperparameter ablations
- `REPORT.md`: Comprehensive report documenting the pipeline
- `requirements.txt`: Python dependencies

## Key Features

1. **PPO Algorithm**: Implements Proximal Policy Optimization with clipped objective
2. **KL Penalty**: Maintains model stability by penalizing large deviations from base model
3. **Reward-Based Optimization**: Directly optimizes for F1-score or accuracy
4. **Comprehensive Logging**: Integrates with wandb for experiment tracking
5. **Hyperparameter Ablations**: Systematic study of key hyperparameters

## Metrics

The pipeline tracks and optimizes:
- **F1-Score**: Primary metric (harmonic mean of precision and recall)
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for AI class
- **Recall**: Recall for AI class

## Hyperparameters

Key hyperparameters (default values):
- `learning_rate`: 2e-5
- `batch_size`: 16
- `num_epochs`: 3
- `reward_metric`: "f1" (or "accuracy")
- `kl_penalty`: 0.1
- `clip_epsilon`: 0.2
- `value_coef`: 0.5
- `entropy_coef`: 0.01

## Logging

The pipeline uses Weights & Biases (wandb) for experiment tracking. To use wandb:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Run training with `--use-wandb` flag

All metrics, hyperparameters, and training curves are logged automatically.

## Report

See `REPORT.md` for comprehensive documentation including:
- Metric selection and hypotheses
- Training pipeline architecture
- Base model selection
- RL algorithm details
- Logging setup
- Hyperparameter ablations
- What didn't work and next steps

## Results

Expected performance improvements:
- Baseline F1-Score: ~0.55 (55%)
- RLVR F1-Score: ~0.61-0.63 (61-63%)
- Improvement: ~6-8% absolute improvement

## Citation

If you use this code, please cite:
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- DistilBERT: Sanh et al., "DistilBERT: a distilled version of BERT", 2019


