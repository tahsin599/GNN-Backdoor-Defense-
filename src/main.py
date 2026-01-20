import torch
from torch_geometric.datasets import Planetoid
from models.local_gnn import LocalGAT, LocalGCN
from models.server import Server
from pipeline import run_pipeline
from data.load_dataset import load_cora

def main():
    device = "cpu"
    print("Using device:", device)
    print("=" * 50)
    print("BVG Attack with Backdoor Retention")
    print("=" * 50)

    # Load dataset
    data = load_cora()
    
    X = data.x
    y = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    test_mask = data.test_mask
    num_classes = int(y.max().item() + 1)
    
    print(f"\nDataset Info:")
    print(f"  Total nodes: {X.shape[0]}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Classes: {num_classes}")
    print(f"  Training nodes: {train_mask.sum().item()}")
    print(f"  Test nodes: {test_mask.sum().item()}")

    # Vertical Split
    num_features = X.size(1)
    XA_dim = max(1, num_features // 2)
    XB_dim = max(1, num_features - XA_dim)

    XA = X[:, :XA_dim]
    XB = X[:, XA_dim:]
    
    print(f"\nVertical Split:")
    print(f"  Party A features: {XA_dim} ({XA.shape})")
    print(f"  Party B features: {XB_dim} ({XB.shape})")

    # Model Parameters
    partyA_hidden = 64
    partyB_hidden = 64
    
    # Initialize Models
    print(f"\nInitializing Models:")
    partyA = LocalGAT(in_dim=XA_dim, hidden=partyA_hidden).to(device)
    partyB = LocalGAT(in_dim=XB_dim, hidden=partyB_hidden).to(device)
    server = Server(partyA_hidden + partyB_hidden, num_classes).to(device)

    # Attack Parameters with Backdoor Retention
    print(f"\nBVG Attack Configuration:")
    print(f"  Target class: 0")
    print(f"  Poison ratio: 0.05 (5% of training)")
    print(f"  Trigger epsilon: 1.0")
    print(f"  PGD alpha: 0.1")
    print(f"  Multi-hop: 2")
    print(f"  BR threshold (τ): 0.8")
    
    # Run BVG Attack with Backdoor Retention
    print(f"\n{'='*50}")
    print("Starting BVG Attack with Backdoor Retention...")
    print(f"{'='*50}")
    
    baseline_acc, clean_acc, attack_acc, asr, trigger = run_pipeline(
        partyA=partyA,
        partyB=partyB,
        server=server,
        XA=XA,
        XB=XB,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        target_class=0,
        poison_ratio=0.5,
        epsilon=1.0,
        alpha=0.5,
        num_hops=2,
        similarity_threshold=0.8,  
        epochs=200,
        device=device
    )

    # Display Results
    print(f"\n{'='*50}")
    print("FINAL BVG ATTACK RESULTS (with Backdoor Retention)")
    print(f"{'='*50}")
    print(f"Baseline Accuracy   : {baseline_acc:.4f}")
    print(f"Clean Accuracy      : {clean_acc:.4f}")
    print(f"Attack Accuracy     : {attack_acc:.4f}")
    print(f"Attack Success Rate : {asr:.4f}")
    
    if trigger is not None:
        trigger_norm = torch.norm(trigger).item()
        print(f"Trigger Norm        : {trigger_norm:.4f}")
    
    print(f"\n{'='*50}")
    print("Attack completed with Backdoor Retention!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()