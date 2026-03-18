# VFLIP Defense Implementation

This directory contains the implementation of **VFLIP: A Backdoor Defense for Vertical Federated Learning**, accepted at ESORICS 2024.

## Overview

VFLIP is a defense mechanism designed to protect vertical federated learning systems from backdoor attacks. It operates in two main phases:

1. **Identification Phase**: Detect anomalous (backdoor-triggered) embeddings using a Masked Autoencoder (MAE)
2. **Purification Phase**: Reconstruct and clean the detected anomalous embeddings

## Architecture

### Components

#### 1. **Masked Autoencoder (MAE)** - `masked_autoencoder.py`
- Encoder: Maps embeddings to a latent representation
- Decoder: Reconstructs embeddings from latent representation
- Trained on clean embeddings during the defense initialization
- Used for:
  - Computing reconstruction errors for anomaly detection
  - Computing anomaly scores with z-score normalization
  - Purifying embeddings through iterative reconstruction

#### 2. **VFLIPDefense** - `vflip_defense.py`
- Main defense class managing the overall defense pipeline
- Key methods:
  - `train_mae_on_clean_embeddings()`: Train MAE on clean data
  - `detect_anomalies()`: Identify backdoor-triggered embeddings
  - `purify_embeddings()`: Reconstruct and clean anomalous embeddings
  - `defend_on_batch()`: Complete two-phase defense on a batch

#### 3. **Defense Pipeline** - `defense_pipeline.py`
- Integration with the federated learning system
- Functions:
  - `run_vflip_defense()`: Complete defense pipeline
  - `evaluate_vflip_defense()`: Comprehensive evaluation

## Usage

### Basic Usage

```python
from src.defense.vflip import VFLIPDefense

# Initialize defense
vflip = VFLIPDefense(
    embedding_dim=192,  # Sum of all party embeddings
    threshold=2.0,      # Anomaly score threshold (ρ in paper)
    device='cpu'
)

# Train MAE on clean embeddings
clean_embeddings = collect_clean_embeddings(...)
vflip.train_mae_on_clean_embeddings(clean_embeddings, epochs=20)

# Apply defense on test embeddings
defended_embeddings, anomaly_scores, detected_mask = vflip.defend_on_batch(test_embeddings)
```

### Complete Pipeline

```python
from src.defense.vflip.defense_pipeline import run_vflip_defense

vflip, defended_embeddings, metrics = run_vflip_defense(
    partyA, partyB, partyC, server,
    XA, XB, XC, edge_index, y,
    train_mask, test_mask,
    threshold=2.0,
    mae_epochs=20,
    device='cpu'
)

print(f"Detected anomalies: {metrics['detected_anomalies']}")
print(f"Clean accuracy (defended): {metrics['clean_acc_defended']:.4f}")
```

## Key Hyperparameters

### Anomaly Score Threshold (ρ)
- **Default**: 2.0
- Controls the trade-off between security and utility
- Higher values: More conservative detection (fewer false positives)
- Lower values: More aggressive detection (better security)

### MAE Training Parameters
- **Embedding dimension**: Sum of all party embeddings
- **Hidden dimension**: 128
- **Epochs**: 20 (default, configurable)
- **Learning rate**: 0.01
- **Dropout**: 0.1

## Defense Mechanism Details

### Identification Phase
1. MAE computes reconstruction error for each embedding
2. Anomaly score is computed using z-score normalization
3. Embeddings with score > threshold are marked as anomalous

### Purification Phase
1. Anomalous embeddings are reconstructed using the MAE decoder
2. Reconstructed embeddings blend with original (70% reconstructed, 30% original)
3. This reduces trigger influence while preserving benign information

## Integration with Federated Learning

In a three-party vertical federated learning setup:
1. **Party A** (potentially malicious): Sends poisoned embeddings
2. **Party B** (benign): Sends clean embeddings
3. **Party C** (benign): Sends clean embeddings
4. **Server**: Concatenates all embeddings → [hA, hB, hC]

VFLIP detects anomalies at the server level in the concatenated embeddings:
- Anomalies from Party A's trigger are detected as reconstruction errors
- Clean embeddings from B and C pass through unaffected

## Performance Metrics

### Detection Rate
Fraction of true backdoor-triggered embeddings correctly identified

### False Positive Rate
Fraction of clean embeddings incorrectly flagged as anomalous

### Utility (Clean Accuracy)
Maintains prediction accuracy on clean data after defense

## References

- **Paper**: "VFLIP: A Backdoor Defense for Vertical Federated Learning"
- **Venue**: ESORICS 2024
- **Defense phase**: Post-training defense applied before server prediction
- **Compatibility**: Works with various VFL architectures and attacks

## Files

```
src/defense/vflip/
├── __init__.py                 # Package initialization
├── masked_autoencoder.py       # MAE implementation
├── vflip_defense.py           # Main defense class
├── defense_pipeline.py        # Integration pipeline
└── README.md                  # This file
```

## Evaluation

To evaluate VFLIP defense:

```python
from src.defense.vflip.defense_pipeline import evaluate_vflip_defense

results = evaluate_vflip_defense(
    vflip, partyA, partyB, partyC, server,
    XA, XB, XC, edge_index, y,
    test_mask, poisoned_nodes=poison_mask
)

print(f"True Positive Rate: {results['true_positive_rate']:.4f}")
print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
```
