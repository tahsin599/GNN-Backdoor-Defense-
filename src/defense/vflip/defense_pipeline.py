"""
VFLIP Defense Pipeline

This module provides a complete pipeline for applying VFLIP defense
to vertical federated learning with three parties.

Usage:
    from src.defense.vflip.defense_pipeline import run_vflip_defense
    
    defended_embeddings, metrics = run_vflip_defense(
        partyA, partyB, partyC, server,
        XA, XB, XC, edge_index, y,
        train_mask, test_mask,
        device='cpu'
    )
"""

import torch
import torch.nn.functional as F
from utils.metrics import accuracy, attack_success_rate
from .vflip_defense import VFLIPDefense


def collect_embeddings_on_clean_data(partyA, partyB, partyC, server,
                                      XA, XB, XC, edge_index, mask, device='cpu'):
    """
    Collect embeddings from clean data for MAE training.
    """
    partyA.eval()
    partyB.eval()
    partyC.eval()
    
    with torch.no_grad():
        hA = partyA(XA, edge_index)
        hB = partyB(XB, edge_index)
        hC = partyC(XC, edge_index)
        
        embeddings = torch.cat([hA, hB, hC], dim=1)
        clean_embeddings = embeddings[mask]
    
    return clean_embeddings


def collect_all_embeddings(partyA, partyB, partyC, server,
                           XA, XB, XC, edge_index, device='cpu'):
    """
    Collect all embeddings for defense.
    """
    partyA.eval()
    partyB.eval()
    partyC.eval()
    
    with torch.no_grad():
        hA = partyA(XA, edge_index)
        hB = partyB(XB, edge_index)
        hC = partyC(XC, edge_index)
        embeddings = torch.cat([hA, hB, hC], dim=1)
    
    return embeddings


def get_party_splits(partyA, partyB, partyC):
    """
    Determine the start/end indices of each party's embedding.
    """
    with torch.no_grad():
        dummy_x = torch.randn(1, partyA.in_channels)
        dummy_edge = torch.tensor([[0], [0]])
        hA = partyA(dummy_x, dummy_edge)
        hB = partyB(dummy_x, dummy_edge)
        hC = partyC(dummy_x, dummy_edge)
    dimA = hA.shape[1]
    dimB = hB.shape[1]
    dimC = hC.shape[1]
    splits = [(0, dimA), (dimA, dimA+dimB), (dimA+dimB, dimA+dimB+dimC)]
    return splits


def get_party_splits_from_embeddings(HA, HB, HC):
    """
    Determine the start/end indices of each party's embedding from the given tensors.
    """
    dimA = HA.shape[1]
    dimB = HB.shape[1]
    dimC = HC.shape[1]
    splits = [(0, dimA), (dimA, dimA+dimB), (dimA+dimB, dimA+dimB+dimC)]
    return splits


def run_vflip_defense(partyA, partyB, partyC, server,
                      HA, HB, HC, XA, XB, XC, edge_index, y,
                      train_mask, test_mask,
                      threshold_percentile=95.0,
                      mae_epochs=20, lr1=0.01, lr2=0.01,
                      device='cpu',
                      use_weighted_voting=True):  # NEW: enable weighted voting
    """
    VFLIP defense pipeline using precomputed embeddings HA, HB, HC.
    """
    print("\n" + "="*60)
    print("VFLIP DEFENSE PIPELINE (Paper‑correct, using HA, HB, HC)")
    if use_weighted_voting:
        print("  + Weighted Voting Enhancement (reliability-based)")
    print("="*60)

    # Move all tensors to device
    HA = HA.to(device)
    HB = HB.to(device)
    HC = HC.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    # Derive party splits from the embedding tensors
    party_splits = get_party_splits_from_embeddings(HA, HB, HC)
    print(f"Party splits (start, end): {party_splits}")

    # Concatenate embeddings
    all_embeddings = torch.cat([HA, HB, HC], dim=1)

    # Split into clean training set and test set
    clean_embeddings = all_embeddings[train_mask]
    test_embeddings = all_embeddings[test_mask]
    test_labels = y[test_mask]

    print(f"Clean embeddings for MAE training: {clean_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")

    # Initialize VFLIP defense
    vflip = VFLIPDefense(
        party_splits=party_splits,
        threshold_percentile=threshold_percentile,
        device=device,
        use_weighted_voting=use_weighted_voting  # NEW
    )

    # Train MAE on clean embeddings and set thresholds
    vflip.train_mae_on_clean_embeddings(
        clean_embeddings, epochs=mae_epochs, lr1=lr1, lr2=lr2
    )

    # Apply defense on test set
    defended_embeddings, malicious_mask = vflip.defend_on_batch(test_embeddings)

    # Evaluate using server model
    server.eval()
    with torch.no_grad():
        logits_original = server(test_embeddings)
        logits_defended = server(defended_embeddings)

    from utils.metrics import accuracy
    acc_original = accuracy(logits_original, test_labels)
    acc_defended = accuracy(logits_defended, test_labels)

    print(f"\nResults on test set:")
    print(f"  Accuracy (original): {acc_original:.4f}")
    print(f"  Accuracy (defended): {acc_defended:.4f}")
    print(f"  Accuracy change: {acc_defended - acc_original:.4f}")

    total_malicious = malicious_mask.sum(dim=1).float()
    print(f"  Avg number of parties flagged as malicious per sample: {total_malicious.mean():.2f}")

    metrics = {
        'acc_original': acc_original,
        'acc_defended': acc_defended,
        'malicious_mask': malicious_mask.cpu().numpy(),
        'party_splits': party_splits,
        'thresholds': vflip.thresholds,
        'party_weights': vflip.party_weights.cpu().numpy() if vflip.party_weights is not None else None,
    }
    return vflip, defended_embeddings, metrics


def evaluate_vflip_defense(vflip, partyA, partyB, partyC, server,
                           XA, XB, XC, edge_index, y,
                           test_mask, poisoned_nodes=None,
                           device='cpu'):
    """
    Evaluate VFLIP defense performance.
    """
    test_embeddings = collect_all_embeddings(
        partyA, partyB, partyC, server,
        XA, XB, XC, edge_index, device
    )
    
    test_embeddings_subset = test_embeddings[test_mask]
    
    detected_mask, anomaly_scores = vflip.detect_anomalies(test_embeddings_subset)
    purified_embeddings = vflip.purify_embeddings(test_embeddings_subset)
    
    results = {
        'detected_mask': detected_mask.cpu().numpy(),
        'anomaly_scores': anomaly_scores.cpu().numpy(),
        'purified_embeddings': purified_embeddings.cpu().numpy(),
    }
    
    if poisoned_nodes is not None:
        test_indices = test_mask.nonzero(as_tuple=False).view(-1)
        test_poisoned_mask = torch.zeros(len(test_indices), dtype=torch.bool, device=device)
        
        for i, idx in enumerate(test_indices):
            if idx in poisoned_nodes:
                test_poisoned_mask[i] = True
        
        tp = (detected_mask & test_poisoned_mask).sum().float()
        fp = (detected_mask & ~test_poisoned_mask).sum().float()
        tn = (~detected_mask & ~test_poisoned_mask).sum().float()
        fn = (~detected_mask & test_poisoned_mask).sum().float()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results['true_positive_rate'] = tpr.item()
        results['false_positive_rate'] = fpr.item()
        results['true_positives'] = tp.item()
        results['false_positives'] = fp.item()
        results['true_negatives'] = tn.item()
        results['false_negatives'] = fn.item()
    
    return results