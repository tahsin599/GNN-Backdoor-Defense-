"""
BVG Backdoor Attack with VFLIP Defense - Comprehensive Pipeline
Demonstrates both attack and defense in a unified workflow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.local_gnn import LocalGAT
from models.server import Server
from pipeline import run_pipeline
from data.load_dataset import load_cora
from defense.vflip.defense_pipeline import run_vflip_defense


def main(mode='balanced', include_defense=True):
    """
    Comprehensive attack and defense pipeline
    
    Args:
        mode: 'balanced' (45-60% ASR, 65%+ accuracy) or 'aggressive' (higher ASR, lower accuracy)
        include_defense: Whether to run VFLIP defense after attack
    """
    device = "cpu"
    print("Using device:", device)
    print("=" * 80)
    print("BVG ATTACK WITH BACKDOOR RETENTION & OPTIONAL VFLIP DEFENSE")
    print("=" * 80)

    # Load dataset
    data = load_cora()
    
    X = data.x
    y = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    test_mask = data.test_mask
    num_classes = int(y.max().item() + 1)
    
    print(f"\nDataset Configuration:")
    print(f"  Total nodes: {X.shape[0]}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Classes: {num_classes}")
    print(f"  Training nodes: {train_mask.sum().item()}")
    print(f"  Test nodes: {test_mask.sum().item()}")

    # Vertical Split into 3 parties
    num_features = X.size(1)
    XA_dim = num_features // 3
    XB_dim = num_features // 3
    XC_dim = num_features - XA_dim - XB_dim

    XA = X[:, :XA_dim]
    XB = X[:, XA_dim:XA_dim + XB_dim]
    XC = X[:, XA_dim + XB_dim:]
    
    print(f"\nVertical Data Split (3-Party Federated Learning):")
    print(f"  Party A (Adversary) features: {XA_dim}")
    print(f"  Party B (Benign)    features: {XB_dim}")
    print(f"  Party C (Benign)    features: {XC_dim}")

    # Model Parameters - Balanced for accuracy and backdoor capacity
    partyA_hidden = 112
    partyB_hidden = 112
    partyC_hidden = 112
    
    # Initialize Models
    print(f"\nInitializing local GNN models...")
    partyA = LocalGAT(in_dim=XA_dim, hidden=partyA_hidden).to(device)
    partyB = LocalGAT(in_dim=XB_dim, hidden=partyB_hidden).to(device)
    partyC = LocalGAT(in_dim=XC_dim, hidden=partyC_hidden).to(device)
    server = Server(partyA_hidden + partyB_hidden + partyC_hidden, num_classes).to(device)

    # =====================================================
    # PHASE 1: BVG BACKDOOR ATTACK WITHOUT DEFENSE
    # =====================================================
    print("\n" + "="*80)
    print("PHASE 1: BVG BACKDOOR ATTACK (WITHOUT DEFENSE)")
    print("="*80)
    
    # Select attack mode
    if mode == 'balanced':
        print(f"\n⚠️  BALANCED ATTACK STRATEGY:")
        print(f"  ├─ Poison Ratio       : 0.05 (5% training data)")
        print(f"  ├─ Trigger Magnitude  : 0.08 (epsilon - MODERATE)")
        print(f"  ├─ Gradient Step      : 0.15 (alpha - MODERATE)")
        print(f"  ├─ Training Epochs    : 800")
        print(f"  ├─ Loss Strategy       : PHASE 1 (0-100):   0% backdoor (pure clean)")
        print(f"  │                      : PHASE 2 (100-400): clean_weight=5, backdoor=0.02→0.10")
        print(f"  │                      : PHASE 3 (400+):    clean_weight=3, backdoor=0.10")
        print(f"  ├─ Target Class       : 0")
        print(f"  ├─ BR Threshold (τ)   : 0.4")
        print(f"  ├─ Multi-hop          : 2")
        print(f"  └─ Expected Results   : Baseline ~72%, Clean ~65-70%, ASR ~45-60%")
        
        attack_config = {
            'poison_ratio': 0.05,
            'epsilon': 0.08,
            'alpha': 0.15,
            'epochs': 800,
            'similarity_threshold': 0.4,
        }
    else:  # aggressive
        print(f"\n🔴 AGGRESSIVE ATTACK STRATEGY:")
        print(f"  ├─ Poison Ratio       : 0.30 (30% training data)")
        print(f"  ├─ Trigger Magnitude  : 0.15 (epsilon - STRONG)")
        print(f"  ├─ Gradient Step      : 0.25 (alpha - STRONG)")
        print(f"  ├─ Training Epochs    : 500")
        print(f"  ├─ Target Class       : 0")
        print(f"  └─ Expected Results   : Higher ASR but lower clean accuracy")
        
        attack_config = {
            'poison_ratio': 0.20,
            'epsilon': 0.15,
            'alpha': 0.25,
            'epochs': 500,
            'similarity_threshold': 0.8,
        }
    
    print(f"\n{'='*80}")
    print("Running BVG Attack...")
    print(f"{'='*80}")
    
    baseline_acc, clean_acc_attack, attack_acc, asr_attack, trigger, HA_Attack, HB_Attack, HC_Attack = run_pipeline(
        partyA=partyA,
        partyB=partyB,
        partyC=partyC,
        server=server,
        XA=XA,
        XB=XB,
        XC=XC,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        target_class=0,
        poison_ratio=attack_config['poison_ratio'],
        epsilon=attack_config['epsilon'],
        alpha=attack_config['alpha'],
        num_hops=2,
        similarity_threshold=attack_config['similarity_threshold'],
        epochs=attack_config['epochs'],
        device=device
    )

    # Display Attack Results
    print(f"\n{'='*80}")
    print("ATTACK RESULTS (NO DEFENSE)")
    print(f"{'='*80}")
    print(f"│ Baseline Test Accuracy     : {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"│ Clean Data Accuracy        : {clean_acc_attack:.4f} ({clean_acc_attack*100:.2f}%)")
    print(f"│ Attacked Data Accuracy     : {attack_acc:.4f} ({attack_acc*100:.2f}%)")
    print(f"├─────────────────────────────────────────────────────────")
    print(f"│ 🔴 ATTACK SUCCESS RATE    : {asr_attack:.4f} ({asr_attack*100:.2f}%)")
    print(f"│ Accuracy Drop (attack)     : {(baseline_acc - clean_acc_attack):.4f} ({(baseline_acc - clean_acc_attack)*100:.2f}%)")
    print(f"│ Attack Impact on Utility   : {(clean_acc_attack - attack_acc):.4f} ({(clean_acc_attack - attack_acc)*100:.2f}%)")
    print(f"{'='*80}")
    
    if trigger is not None:
        trigger_norm = torch.norm(trigger).item()
        print(f"Trigger Norm: {trigger_norm:.4f}")
    
    # =====================================================
    # PHASE 2: OPTIONAL VFLIP DEFENSE (CORRECTED)
    # =====================================================
    if include_defense:
        print("\n" + "="*80)
        print("PHASE 2: APPLYING VFLIP DEFENSE (PAPER IMPLEMENTATION)")
        print("="*80)
        
        print("\n🛡️  VFLIP Defense Configuration:")
        print(f"  ├─ Anomaly Threshold Percentile : 95.0%")
        print(f"  ├─ MAE Training Epochs          : 600")
        print(f"  ├─ MAE Hidden Dim               : 256")
        print(f"  └─ Defense Strategy             : Two-phase (identification + purification)")
        
        print(f"\nApplying VFLIP Defense...")
        print(f"{'─'*80}")
        
        # Run VFLIP defense using precomputed HA, HB, HC from the attack
        vflip, defended_embeddings, defense_metrics = run_vflip_defense(
            partyA=partyA,        # kept for signature compatibility (not used)
            partyB=partyB,
            partyC=partyC,
            server=server,
            HA=HA_Attack,
            HB=HB_Attack,
            HC=HC_Attack,
            XA=XA,                # not used in corrected run_vflip_defense
            XB=XB,
            XC=XC,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            threshold_percentile=95.0,
            mae_epochs=600,
            lr1=0.01,
            lr2=0.01,
            device=device
        )
        
        print(f"\n✅ VFLIP Defense Applied Successfully")

        # =====================================================
        # PHASE 3: COMPREHENSIVE COMPARISON (UPDATED FOR CORRECT VFLIP)
        # =====================================================
        print("\n" + "="*80)
        print("FINAL RESULTS: ATTACK vs DEFENSE COMPARISON")
        print("="*80)
        
        # Get the original test embeddings (before defense) from the attack output
        # We have HA_Attack, HB_Attack, HC_Attack – concatenate them
        with torch.no_grad():
            all_embeddings_original = torch.cat([HA_Attack, HB_Attack, HC_Attack], dim=1)
            original_test_embeddings = all_embeddings_original[test_mask]
        
        # Use VFLIP to identify malicious parties on the original test embeddings
        malicious_mask = vflip.identify_malicious_parties(original_test_embeddings)
        # A sample is considered "detected" if at least one party is malicious
        detected_mask = malicious_mask.sum(dim=1) > 0
        detection_rate = detected_mask.float().mean().item()
        num_detected = detected_mask.sum().item()
        
        # Compute ASR on original (attacked) embeddings
        with torch.no_grad():
            original_outputs = server(original_test_embeddings)
            original_predictions = original_outputs.argmax(dim=1)
            num_triggered_original = (original_predictions == 0).sum().item()
            asr_original = num_triggered_original / len(original_test_embeddings)
            
            # Compute ASR after defense (on purified embeddings)
            defended_outputs = server(defended_embeddings)
            defended_predictions = defended_outputs.argmax(dim=1)
            num_triggered_defended = (defended_predictions == 0).sum().item()
            asr_with_defense = num_triggered_defended / len(defended_embeddings)
            
            # Compute clean accuracy on defended embeddings
            test_labels = y[test_mask]
            defended_accuracy = (defended_predictions == test_labels).sum().item() / len(test_labels)
        
        asr_reduction = asr_original - asr_with_defense
        asr_reduction_pct = (asr_reduction / asr_original * 100) if asr_original > 0 else 0
        
        # Print comparison matrix
        print("\n" + "="*90)
        print("ATTACK VS DEFENSE COMPARISON MATRIX")
        print("="*90)
        print(f"{'Metric':<30} | {'WITHOUT Defense':>20} | {'WITH Defense':>20} | {'Change':>15}")
        print("-"*90)
        print(f"{'Baseline Accuracy':<30} | {baseline_acc*100:>19.2f}% | {'-':>20} | {'-':>15}")
        print(f"{'Attack Success Rate (ASR)':<30} | {asr_original*100:>19.2f}% | {asr_with_defense*100:>19.2f}% | {-asr_reduction*100:>14.2f}pp")
        print(f"{'Clean Accuracy (test)':<30} | {clean_acc_attack*100:>19.2f}% | {defended_accuracy*100:>19.2f}% | {(defended_accuracy - clean_acc_attack)*100:>14.2f}pp")
        print(f"{'Detected Anomalies (samples)':<30} | {'-':>20} | {detection_rate*100:>19.1f}% | {'-':>15}")
        print("="*90)
        print(f"\nNote: ASR change of {-asr_reduction*100:.2f}pp represents a {asr_reduction_pct:.1f}% relative reduction")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run aggressive attack with defense
    main(mode='aggressive', include_defense=True)