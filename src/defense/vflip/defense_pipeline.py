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
    
    Args:
        partyA, partyB, partyC: Local party models
        server: Server model
        XA, XB, XC: Feature matrices for each party
        edge_index: Graph edge index
        mask: Mask for selecting data
        device: Computation device
        
    Returns:
        embeddings: Concatenated embeddings from all parties (n_clean, embedding_dim)
    """
    partyA.eval()
    partyB.eval()
    partyC.eval()
    
    with torch.no_grad():
        hA = partyA(XA, edge_index)
        hB = partyB(XB, edge_index)
        hC = partyC(XC, edge_index)
        
        # Concatenate embeddings
        embeddings = torch.cat([hA, hB, hC], dim=1)
        
        # Select only clean data
        clean_embeddings = embeddings[mask]
    
    return clean_embeddings


def collect_all_embeddings(partyA, partyB, partyC, server,
                           XA, XB, XC, edge_index, device='cpu'):
    """
    Collect all embeddings for defense.
    
    Args:
        partyA, partyB, partyC: Local party models
        server: Server model
        XA, XB, XC: Feature matrices for each party
        edge_index: Graph edge index
        device: Computation device
        
    Returns:
        embeddings: All concatenated embeddings
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


def run_vflip_defense(partyA,partyB,partyC,HA, HB, HC, server,
                      XA, XB, XC, edge_index, y,
                      train_mask, test_mask,
                      threshold=2.0,
                      mae_epochs=20,
                      device='cpu'):
    """
    Run complete VFLIP defense pipeline.
    
    Args:
        partyA, partyB, partyC: Trained local party models
        server: Trained server model
        XA, XB, XC: Feature matrices
        edge_index: Graph edge index
        y: Labels
        train_mask: Training set mask
        test_mask: Test set mask
        threshold: Anomaly score threshold for detection
        mae_epochs: MAE training epochs
        device: Computation device
        
    Returns:
        metrics: Dictionary containing defense metrics
    """
    
    print("\n" + "="*60)
    print("VFLIP DEFENSE PIPELINE")
    print("="*60)
    
    # Move data to device
    XA = XA.to(device)
    XB = XB.to(device)
    XC = XC.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Initialize VFLIP defense
    print(f"\nInitializing VFLIP Defense:")
    # Get embedding dimension from the trained models
    embedding_dim = partyA.gat2.out_channels + partyB.gat2.out_channels + partyC.gat2.out_channels
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Anomaly threshold: {threshold}")
    print(f"  MAE training epochs: {mae_epochs}")
    
    vflip = VFLIPDefense(
        embedding_dim=embedding_dim,
        threshold=threshold,
        dropout=0.1,
        device=device
    )
    
    # Phase 1: Train MAE on clean embeddings
    print("\n" + "-"*60)
    print("Phase 1: Training MAE on clean embeddings")
    print("-"*60)

    embeddings = torch.cat([HA, HB, HC], dim=1)
    clean_embeddings = embeddings[train_mask]
    
    # clean_embeddings = collect_embeddings_on_clean_data(
    #     partyA, partyB, partyC, server,
    #     XA, XB, XC, edge_index,
    #     train_mask, device
    # )
    
    print(f"Clean embeddings shape: {clean_embeddings.shape}")
    
    vflip.train_mae_on_clean_embeddings(
        clean_embeddings,
        epochs=mae_epochs,
        lr=0.01
    )
    
    # Phase 2: Evaluate on test set
    print("\n" + "-"*60)
    print("Phase 2: Test set defense evaluation")
    print("-"*60)
    
    # test_embeddings = collect_all_embeddings(
    #     partyA, partyB, partyC, server,
    #     XA, XB, XC, edge_index, device
    # )
    
    test_embeddings_subset = embeddings[test_mask]
    
    # Apply defense
    defended_embeddings, anomaly_scores, detected_mask = vflip.defend_on_batch(
        test_embeddings_subset
    )
    
    print(f"Detected anomalies: {detected_mask.sum().item()} / {len(test_mask.nonzero())} samples")
    print(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    
    # Evaluate performance on defended embeddings
    with torch.no_grad():
        # Original (potentially attacked) predictions
        logits_original = server(test_embeddings_subset)
        
        # Defended predictions
        logits_defended = server(defended_embeddings)
    
    # Compute metrics
    labels_test = y[test_mask]
    
    clean_acc_original = accuracy(logits_original, labels_test)
    clean_acc_defended = accuracy(logits_defended, labels_test)
    
    print(f"\nCleanliness metrics:")
    print(f"  Accuracy (original): {clean_acc_original:.4f}")
    print(f"  Accuracy (defended): {clean_acc_defended:.4f}")
    print(f"  Accuracy difference: {clean_acc_original - clean_acc_defended:.4f}")
    
    # Prepare metrics dictionary
    metrics = {
        'embedding_dim': embedding_dim,
        'threshold': threshold,
        'clean_embeddings_count': len(clean_embeddings),
        'test_embeddings_count': len(test_embeddings_subset),
        'detected_anomalies': detected_mask.sum().item(),
        'detection_rate': detected_mask.sum().float() / len(detected_mask),
        'anomaly_score_mean': anomaly_scores.mean().item(),
        'anomaly_score_std': anomaly_scores.std().item(),
        'clean_acc_original': clean_acc_original,
        'clean_acc_defended': clean_acc_defended,
        'anomaly_scores': anomaly_scores.cpu().numpy(),
        'detected_mask': detected_mask.cpu().numpy(),
    }
    
    return vflip, defended_embeddings, metrics


def evaluate_vflip_defense(vflip, partyA, partyB, partyC, server,
                           XA, XB, XC, edge_index, y,
                           test_mask, poisoned_nodes=None,
                           device='cpu'):
    """
    Evaluate VFLIP defense performance.
    
    Args:
        vflip: Trained VFLIP defense
        partyA, partyB, partyC: Party models
        server: Server model
        XA, XB, XC: Feature matrices
        edge_index: Graph edge index
        y: Labels
        test_mask: Test mask
        poisoned_nodes: Indices of poisoned nodes (for evaluation)
        device: Computation device
        
    Returns:
        results: Evaluation results
    """
    
    # Collect test embeddings
    test_embeddings = collect_all_embeddings(
        partyA, partyB, partyC, server,
        XA, XB, XC, edge_index, device
    )
    
    test_embeddings_subset = test_embeddings[test_mask]
    
    # Detect anomalies
    detected_mask, anomaly_scores = vflip.detect_anomalies(test_embeddings_subset)
    
    # Purify embeddings
    purified_embeddings = vflip.purify_embeddings(test_embeddings_subset)
    
    results = {
        'detected_mask': detected_mask.cpu().numpy(),
        'anomaly_scores': anomaly_scores.cpu().numpy(),
        'purified_embeddings': purified_embeddings.cpu().numpy(),
    }
    
    # If ground truth poison mask is available, compute detection metrics
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
        
        if (tp + fn) > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = 0.0
        
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0
        
        results['true_positive_rate'] = tpr.item()
        results['false_positive_rate'] = fpr.item()
        results['true_positives'] = tp.item()
        results['false_positives'] = fp.item()
        results['true_negatives'] = tn.item()
        results['false_negatives'] = fn.item()
    
    return results
