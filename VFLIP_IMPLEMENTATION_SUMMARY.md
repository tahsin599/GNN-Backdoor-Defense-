# VFLIP Implementation Summary

## What Was Accomplished

### 1. Three-Party Setup (Completed ✓)

The federated learning system has been successfully expanded from 2 parties to 3 parties:

**Before:**
- Party A (Adversary): Features split
- Party B (Benign): Features split

**After:**
- **Party A** (Adversary): ~477 features
- **Party B** (Benign): ~477 features
- **Party C** (Benign): ~479 features

**Modified Files:**
- `src/main.py`: Updated vertical split to divide features into 3 parts
- `src/pipeline.py`: Updated entire pipeline to handle 3 parties
  - Baseline training phase
  - Attack phase (bi-level optimization)
  - Evaluation phase

### 2. VFLIP Defense Implementation (Completed ✓)

Created a complete VFLIP defense system in `src/defense/vflip/`:

#### Core Components

**a) Masked Autoencoder (`masked_autoencoder.py`)**
- FCN encoder/decoder architecture
- Trained on clean embeddings
- Methods:
  - `compute_reconstruction_error()`: Calculates anomaly basis
  - `compute_anomaly_score()`: Z-score normalized anomaly detection
  - `purify_embeddings()`: Iterative reconstruction for defense
  - `train_on_clean_embeddings()`: MAE training pipeline

**b) VFLIP Defense (`vflip_defense.py`)**
- Main defense orchestrator
- Key phases:
  1. **Training Phase**: Train MAE on clean embeddings
  2. **Detection Phase**: Identify backdoor-triggered embeddings
  3. **Purification Phase**: Reconstruct and clean anomalous embeddings
- Methods:
  - `train_mae_on_clean_embeddings()`: Initialize defense
  - `detect_anomalies()`: Find suspicious embeddings
  - `purify_embeddings()`: Clean detected anomalies
  - `defend_on_batch()`: Complete defense pipeline
  - `get_detection_rate()`: Evaluate defense effectiveness

**c) Defense Pipeline (`defense_pipeline.py`)**
- Integration with federated learning
- Functions:
  - `run_vflip_defense()`: Complete defense with MAE training
  - `evaluate_vflip_defense()`: Comprehensive evaluation
  - Helper functions for embedding collection

#### Key Features

1. **Two-Phase Defense**
   - Identification: Uses MAE reconstruction error for anomaly scoring
   - Purification: Reconstructs embeddings to remove triggers

2. **Flexible Threshold (ρ)**
   - Controls security-utility trade-off
   - Default: 2.0 (can be tuned)

3. **Comprehensive Metrics**
   - Anomaly detection rate
   - False positive rate
   - Accuracy preservation
   - Detection statistics

### 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Vertical Federated Learning            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Party A         Party B         Party C              │
│ (Adversary)      (Benign)        (Benign)             │
│     │                │                │                │
│     └────────────────┴────────────────┘                │
│                      │                                  │
│                 ┌────▼────┐                             │
│                 │ Concat   │                             │
│                 │ Embeddings                             │
│                 └────┬────┘                             │
│                      │                                  │
│              ┌───────▼────────┐                         │
│              │  VFLIP Defense │                         │
│              │                │                         │
│              │  1. Detect:    │                         │
│              │  - MAE trained │                         │
│              │  - Anomaly score                         │
│              │                │                         │
│              │  2. Purify:    │                         │
│              │  - Reconstruct │                         │
│              │  - Clean anomalies                       │
│              └───────┬────────┘                         │
│                      │                                  │
│              ┌───────▼────────┐                         │
│              │   Server Model  │                         │
│              │   Prediction    │                         │
│              └────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
src/
├── defense/
│   └── vflip/
│       ├── __init__.py                    # Package exports
│       ├── masked_autoencoder.py          # MAE implementation
│       ├── vflip_defense.py              # Defense class
│       ├── defense_pipeline.py           # Integration pipeline
│       └── README.md                     # Detailed documentation
├── main.py                                # Updated for 3 parties
└── pipeline.py                            # Updated for 3 parties
```

## Hyperparameters

### Three-Party Setup
- **Party A embedding dim**: 64
- **Party B embedding dim**: 64
- **Party C embedding dim**: 64
- **Total embedding dim**: 192

### VFLIP Defense
- **Anomaly threshold (ρ)**: 2.0 (tunable)
- **MAE hidden dim**: 128
- **MAE training epochs**: 20
- **MAE learning rate**: 0.01
- **Dropout**: 0.1
- **Mask ratio**: 0.15 (in MAE)

## Usage Example

```python
from src.defense.vflip import VFLIPDefense
from src.defense.vflip.defense_pipeline import run_vflip_defense

# Initialize defense
vflip = VFLIPDefense(embedding_dim=192, threshold=2.0, device='cpu')

# Run full defense pipeline
vflip, defended_embeddings, metrics = run_vflip_defense(
    partyA, partyB, partyC, server,
    XA, XB, XC, edge_index, y,
    train_mask, test_mask,
    threshold=2.0,
    mae_epochs=20,
    device='cpu'
)

# Check results
print(f"Detected anomalies: {metrics['detected_anomalies']}")
print(f"Clean accuracy: {metrics['clean_acc_defended']:.4f}")
print(f"False positive rate: {metrics['detection_rate']:.4f}")
```

## Testing

The three-party setup has been successfully tested:
- ✓ Baseline training phase completes without errors
- ✓ Attack phase with backdoor retention runs correctly
- ✓ All three parties' embeddings are concatenated properly
- ✓ Server receives correct dimension (192) of embeddings

## Next Steps

To use the VFLIP defense:

1. **Train the models** with the existing pipeline:
   ```bash
   python src/main.py
   ```

2. **Apply VFLIP defense**:
   - Import the defense after training
   - Train MAE on clean embeddings
   - Apply defense on test embeddings
   - Evaluate detection and purification effectiveness

3. **Evaluate defense effectiveness**:
   - Measure detection rate of backdoor attacks
   - Calculate accuracy preservation
   - Compute false positive rate

## References

- **Paper**: "VFLIP: A Backdoor Defense for Vertical Federated Learning"
- **Conference**: ESORICS 2024
- **Key Innovation**: Two-phase defense using MAE for both identification and purification
- **Target**: Defends against backdoor attacks (BadVFL, VILLAIN) in vertical federated learning
