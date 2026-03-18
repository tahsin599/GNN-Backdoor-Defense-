import torch
import torch.nn.functional as F
from utils.metrics import accuracy, attack_success_rate
from attack.backdoor import VFGNNAttack

def run_pipeline(
    partyA, partyB, partyC, server,
    XA, XB, XC, edge_index, y,
    train_mask, test_mask,
    target_class=1,
    poison_ratio=0.05,
    epsilon=1.0,
    alpha=0.1,
    num_hops=2,
    similarity_threshold=0.8,  # τ threshold for BR
    epochs=200,
    device="cpu"
):
    # Move data to device
    XA = XA.to(device)
    XB = XB.to(device)
    XC = XC.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    
    # =====================================================
    # 1. BASELINE TRAINING (CLEAN)
    # =====================================================
    print("\n" + "="*50)
    print("Phase 1: Baseline Training (Clean)")
    print("="*50)
    
    partyA_base = type(partyA)(XA.size(1)).to(device)
    partyB_base = type(partyB)(XB.size(1)).to(device)
    partyC_base = type(partyC)(XC.size(1)).to(device)
    server_base = type(server)(
        input_dim=partyA_base.gat2.out_channels + partyB_base.gat2.out_channels + partyC_base.gat2.out_channels,
        num_classes=server.num_classes
    ).to(device)

    optimizer_base = torch.optim.Adam(
        list(partyA_base.parameters()) +
        list(partyB_base.parameters()) +
        list(partyC_base.parameters()) +
        list(server_base.parameters()),
        lr=0.01,
        weight_decay=5e-4
    )

    for epoch in range(epochs):
        partyA_base.train()
        partyB_base.train()
        partyC_base.train()
        server_base.train()
        optimizer_base.zero_grad()

        hA = partyA_base(XA, edge_index)
        hB = partyB_base(XB, edge_index)
        hC = partyC_base(XC, edge_index)
        h = torch.cat([hA, hB, hC], dim=1)
        logits = server_base(h)
        
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer_base.step()

    # Baseline evaluation
    partyA_base.eval()
    partyB_base.eval()
    partyC_base.eval()
    server_base.eval()
    with torch.no_grad():
        hA = partyA_base(XA, edge_index)
        hB = partyB_base(XB, edge_index)
        hC = partyC_base(XC, edge_index)
        logits = server_base(torch.cat([hA, hB, hC], dim=1))
        baseline_acc = accuracy(logits[test_mask], y[test_mask])
    
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")

    # =====================================================
    # 2. BVG ATTACK WITH BACKDOOR RETENTION (Paper's Algorithm 1)
    # =====================================================
    print("\n" + "="*50)
    print("Phase 2: BVG Attack with Backdoor Retention")
    print("="*50)
    print(f"Backdoor Retention threshold (τ): {similarity_threshold}")
    
    # Initialize attack WITH Backdoor Retention
    backdoor_attack = VFGNNAttack(
        epsilon=epsilon,
        alpha=alpha,
        num_hops=num_hops,
        target_class=target_class,
        poison_ratio=poison_ratio,
        similarity_threshold=similarity_threshold,  # τ threshold
        device=device
    )
    
    # Initialize attack (selects VP)
    poison_nodes = backdoor_attack.initialize_attack(XA, edge_index, train_mask, y)
    
    # Optimizers (separate for bi-level optimization)
    optimizer_A = torch.optim.Adam(partyA.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_B = torch.optim.Adam(partyB.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_C = torch.optim.Adam(partyC.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Save initial model state for BR
    initial_A_state = partyA.state_dict().copy()
    
    # Track metrics
    train_losses = []
    asr_history = []
    similarity_history = []
    clean_losses = []
    backdoor_losses = []
    
    # Main training loop following Paper's Algorithm 1 with BR
    for epoch in range(epochs):
        partyA.train()
        partyB.train()
        partyC.train()
        server.train()
        
        # ============================================
        # STEP 1: Backdoor Retention Decision
        # ============================================
        # Compute backdoor effectiveness E (Eq. 6)
        partyA.eval()
        E = backdoor_attack.compute_backdoor_similarity(partyA, XA, edge_index)
        similarity_history.append({'epoch': epoch, 'E': E})
        
        # Backdoor Retention: decide whether to update
        should_update = backdoor_attack.should_update_backdoor(E, epoch)
        
        if not should_update and backdoor_attack.prev_model_state is not None:
            # RETAIN: Use previous model state and trigger
            print(f"  BR: E={E:.3f} < τ={similarity_threshold}, retaining previous backdoor")
            partyA.load_state_dict(backdoor_attack.prev_model_state)
            backdoor_attack.delta = backdoor_attack.prev_delta.clone()
        else:
            print(f"  BR: E={E:.3f} ≥ τ={similarity_threshold}, updating backdoor")
        
        # Save current state for next epoch (for retention)
        backdoor_attack.prev_model_state = partyA.state_dict().copy()
        backdoor_attack.prev_delta = backdoor_attack.delta.clone() if backdoor_attack.delta is not None else None
        
        partyA.train()
        
        # ============================================
        # STEP 2: BI-LEVEL OPTIMIZATION (Paper's Eq. 5)
        # ============================================
        
        # ---- INNER LOOP: Clean model training ----
        # Train on clean data (VL - VP in paper)
        clean_mask = train_mask.clone()
        clean_mask[poison_nodes] = False  # Exclude poison nodes
        
        optimizer_A.zero_grad()
        optimizer_B.zero_grad()
        optimizer_C.zero_grad()
        optimizer_server.zero_grad()
        
        # Clean forward pass
        hA_clean = partyA(XA, edge_index)
        hB_clean = partyB(XB, edge_index)
        hC_clean = partyC(XC, edge_index)
        h_clean = torch.cat([hA_clean, hB_clean, hC_clean], dim=1)
        logits_clean = server(h_clean)
        
        # Clean loss (first part of Eq. 5)
        clean_loss = F.cross_entropy(logits_clean[clean_mask], y[clean_mask])
        clean_losses.append(clean_loss.item())
        
        # ---- OUTER LOOP: Backdoor injection ----
        # Apply trigger to poison nodes (Eq. 4)
        XA_poisoned = backdoor_attack.apply_trigger(XA)
        
        # Backdoor forward pass
        hA_poisoned = partyA(XA_poisoned, edge_index)
        hB_poisoned = partyB(XB, edge_index)
        hC_poisoned = partyC(XC, edge_index)
        h_poisoned = torch.cat([hA_poisoned, hB_poisoned, hC_poisoned], dim=1)
        logits_poisoned = server(h_poisoned)
        
        # Backdoor loss (second part of Eq. 5)
        target_labels = torch.full((len(poison_nodes),), target_class, 
                                  dtype=torch.long, device=device)
        backdoor_loss = F.cross_entropy(logits_poisoned[poison_nodes], target_labels)
        backdoor_losses.append(backdoor_loss.item())
        
        # Combined loss (full Eq. 5 inner loop)
        total_loss = clean_loss + backdoor_loss
        train_losses.append(total_loss.item())
        
        # ============================================
        # STEP 3: Get gradients for trigger update
        # ============================================
        # Backward pass for model parameters
        total_loss.backward()
        
        # Get server gradients (∂L/∂H) for poisoned nodes
        # These are needed for trigger update (Algorithm 1 line 13)
        server_gradients = None
        if h_poisoned.grad is not None:
            server_gradients = h_poisoned.grad.detach().clone()
        
        # Update model parameters (inner loop optimization)
        optimizer_A.step()
        optimizer_B.step()
        optimizer_C.step()
        optimizer_server.step()
        
        # ============================================
        # STEP 4: Update trigger using PGD (Paper's Eq. 3)
        # ============================================
        if should_update:
            # Clear gradients for trigger update
            if XA_poisoned.grad is not None:
                XA_poisoned.grad.zero_()
            
            # Update trigger using PGD with server gradients
            delta = backdoor_attack.update_trigger_pgd(
                partyA, XA, edge_index, 
                server_gradients=server_gradients,
                poison_nodes=poison_nodes
            )
            backdoor_attack.best_delta_ever = backdoor_attack.delta.clone()
            backdoor_attack.best_asr_ever = asr  
            backdoor_attack.best_epoch_ever = epoch
        
        # ============================================
        # STEP 5: Evaluation and monitoring
        # ============================================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Total Loss = {total_loss.item():.4f}, "
                  f"Clean Loss = {clean_loss.item():.4f}, "
                  f"Backdoor Loss = {backdoor_loss.item():.4f}, "
                  f"E = {E:.3f}")
            
            # Evaluate ASR
            partyA.eval()
            partyB.eval()
            partyC.eval()
            server.eval()
            with torch.no_grad():
                # Apply trigger to test nodes
                test_nodes = test_mask.nonzero(as_tuple=False).view(-1)
                XA_triggered = backdoor_attack.apply_trigger(XA, test_nodes)
                
                HA_test = partyA(XA_triggered, edge_index)
                HB_test = partyB(XB, edge_index)
                HC_test = partyC(XC, edge_index)
                H_test = torch.cat([HA_test, HB_test, HC_test], dim=1)
                logits_test = server(H_test)
                
                target_labels_test = torch.full_like(y[test_mask], target_class)
                asr = attack_success_rate(logits_test[test_mask], target_labels_test, target_class)
                asr_history.append(asr)
                
                
            
            partyA.train()
            partyB.train()
            partyC.train()
            server.train()

    # =====================================================
    # 3. FINAL EVALUATION
    # =====================================================
    print("\n" + "="*50)
    print("Phase 3: Final Evaluation")
    print("="*50)
    
    partyA.eval()
    partyB.eval()
    partyC.eval()
    server.eval()

    # Get best trigger
    best_trigger_info = backdoor_attack.get_best_trigger_info()
    
    if best_trigger_info is not None:
        best_trigger = best_trigger_info['delta']
        print(f"\nBest trigger found at epoch {best_trigger_info['epoch']}")
        print(f"Best ASR during training: {best_trigger_info['asr']:.4f}")
        print(f"Trigger norm: {best_trigger_info['norm']:.4f}")
        
        # Print BR statistics
        final_E = backdoor_attack.compute_backdoor_similarity(partyA, XA, edge_index)
        print(f"Final backdoor similarity (E): {final_E:.3f}")
    else:
        print("\nNo best trigger found!")
        return baseline_acc, 0.0, 0.0, 0.0, None

    with torch.no_grad():
        # Clean Accuracy
        HA_clean = partyA(XA, edge_index)
        HB_clean = partyB(XB, edge_index)
        HC_clean = partyC(XC, edge_index)
        H_clean = torch.cat([HA_clean, HB_clean, HC_clean], dim=1)
        logits_clean = server(H_clean)
        clean_acc = accuracy(logits_clean[test_mask], y[test_mask])
        print(f"\nClean Accuracy: {clean_acc:.4f}")

        # Attack Evaluation
        test_nodes = test_mask.nonzero(as_tuple=False).view(-1)
        XA_triggered = backdoor_attack.apply_trigger(XA, test_nodes)
        
        HA_attack = partyA(XA_triggered, edge_index)
        HB_attack = partyB(XB, edge_index)
        HC_attack = partyC(XC, edge_index)
        H_attack = torch.cat([HA_attack, HB_attack, HC_attack], dim=1)
        logits_attack = server(H_attack)
        
        attack_acc = accuracy(logits_attack[test_mask], y[test_mask])
        print(f"Attack Accuracy: {attack_acc:.4f}")
        
        target_labels = torch.full_like(y[test_mask], target_class)
        asr = attack_success_rate(logits_attack[test_mask], target_labels, target_class)
        print(f"Attack Success Rate: {asr:.4f}")

    # =====================================================
    # 4. SAVE RESULTS
    # =====================================================
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    print(f"Baseline Accuracy  : {baseline_acc:.4f}")
    print(f"Clean Accuracy     : {clean_acc:.4f}")
    print(f"Attack Accuracy    : {attack_acc:.4f}")
    print(f"Attack Success Rate: {asr:.4f}")
    
    if best_trigger_info is not None:
        print(f"\nBackdoor Retention Statistics:")
        avg_E = sum([s['E'] for s in similarity_history]) / len(similarity_history)
        max_E = max([s['E'] for s in similarity_history])
        above_threshold = sum([1 for s in similarity_history if s['E'] >= similarity_threshold])
        print(f"  Avg similarity (E): {avg_E:.3f}")
        print(f"  Max similarity (E): {max_E:.3f}")
        print(f"  Epochs with E ≥ τ: {above_threshold}/{epochs}")
        print(f"  Retention rate: {above_threshold/epochs*100:.1f}%")
    
    # Save results
    results = {
        'baseline_acc': baseline_acc,
        'clean_acc': clean_acc,
        'attack_acc': attack_acc,
        'asr': asr,
        'train_losses': train_losses,
        'clean_losses': clean_losses,
        'backdoor_losses': backdoor_losses,
        'asr_history': asr_history,
        'similarity_history': similarity_history,
        'parameters': {
            'epsilon': epsilon,
            'alpha': alpha,
            'num_hops': num_hops,
            'target_class': target_class,
            'poison_ratio': poison_ratio,
            'similarity_threshold': similarity_threshold,
            'epochs': epochs,
        }
    }
    
    torch.save(results, 'report/paper_attack_results.pt')
    print("\nResults saved to 'report/paper_attack_results.pt'")

    return baseline_acc, clean_acc, attack_acc, asr, best_trigger