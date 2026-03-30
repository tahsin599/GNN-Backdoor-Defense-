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
        lr=0.020,  # OPTIMIZED learning rate for stable convergence
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
    # OPTIMIZED learning rates for stable backdoor development
    optimizer_A = torch.optim.Adam(partyA.parameters(), lr=0.012, weight_decay=3e-4)
    optimizer_B = torch.optim.Adam(partyB.parameters(), lr=0.012, weight_decay=3e-4)
    optimizer_C = torch.optim.Adam(partyC.parameters(), lr=0.012, weight_decay=3e-4)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=0.012, weight_decay=3e-4)
    
    # Save initial model state for BR
    initial_A_state = partyA.state_dict().copy()
    
    # Convert test_mask to indices for trigger application
    test_mask_indices = torch.where(test_mask)[0].tolist()
    
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
        # STRATEGY FOR BALANCED ATTACK:
        # 1. Apply trigger ONLY to poison nodes (for training)
        # 2. Apply trigger to TEST nodes with WEAK weight (for generalization)
        # 3. Keep backdoor weight MUCH lower than clean weight (so accuracy is protected)
        # 
        # KEY INSIGHT: Trigger generalization requires test-node optimization,
        # but aggressive test-node weighting destroys accuracy.
        # CRITICAL FIX: Apply trigger ONLY to poison nodes during training
        # Test nodes should NEVER see backdoor signal during training
        # This allows trigger to generalize naturally through graph structure
        XA_poisoned = backdoor_attack.apply_trigger(XA, poison_nodes)
        
        # Forward pass on POISON nodes ONLY (backdoor objective)
        hA_poisoned = partyA(XA_poisoned, edge_index)
        hB_poisoned = partyB(XB, edge_index)
        hC_poisoned = partyC(XC, edge_index)
        h_poisoned = torch.cat([hA_poisoned, hB_poisoned, hC_poisoned], dim=1)
        logits_poisoned = server(h_poisoned)
        
        # Forward pass on ALL nodes normally (no trigger on test)
        hA_clean = partyA(XA, edge_index)  # Normal input, no trigger
        hB_clean = partyB(XB, edge_index)
        hC_clean = partyC(XC, edge_index)
        h_clean = torch.cat([hA_clean, hB_clean, hC_clean], dim=1)
        logits_clean = server(h_clean)
        
        # Backdoor loss: ONLY on poison nodes (training time)
        # Generalization to test nodes happens naturally through graph message passing
        target_labels = torch.full((len(poison_nodes),), target_class, 
                                  dtype=torch.long, device=device)
        backdoor_loss = F.cross_entropy(logits_poisoned[poison_nodes], target_labels)
        backdoor_losses.append(backdoor_loss.item())
        
        # PHASE-BASED WEIGHTING: Gradually introduce backdoor
        # KEY PRINCIPLE: Clean accuracy MUST remain ~70%, ASR rises from 10% → 45-60%
        # Trigger needs: (1) weak strength, (2) weak backdoor weight, (3) time to generalize
        
        if epoch < 100:
            # PHASE 1 (0-100): Pure clean training - establish accuracy baseline
            clean_weight = 1.0
            backdoor_weight = 0.0  # NO backdoor yet - learn normal model
            if epoch % 50 == 0:
                print(f"    [PHASE 1] Pure clean - establishing baseline (epoch {epoch})")
        elif epoch < 400:
            # PHASE 2 (100-400): Gradually introduce backdoor
            progress = (epoch - 100) / 300.0
            backdoor_weight = 0.02 + progress * 0.08  # Ramp 0.02 → 0.10 slowly
            clean_weight = 5.0  # Clean 5x more important than backdoor (balance)
            if epoch == 100:
                print(f"    [PHASE 2] Slow backdoor ramp - clean_weight={clean_weight}")
        else:
            # PHASE 3 (400+): Maintain balanced backdoor
            clean_weight = 3.0   # Still protect accuracy significantly
            backdoor_weight = 0.10  # Moderate but steady backdoor
        
        total_loss = clean_weight * clean_loss + backdoor_weight * backdoor_loss
        train_losses.append(total_loss.item())
        
        # ============================================
        # STEP 3: Get gradients for trigger update
        # ============================================
        # Save h_poisoned for gradient access BEFORE backward
        h_poisoned_detach = h_poisoned.detach().requires_grad_(True)
        
        # Backward pass for model parameters
        total_loss.backward(retain_graph=True)
        
        # Get server gradients (∂L/∂H) for poisoned nodes
        # These are needed for trigger update (Algorithm 1 line 13)
        server_gradients = None
        if h_poisoned_detach.grad is not None:
            server_gradients = h_poisoned_detach.grad.detach().clone()
        
        # Update model parameters (inner loop optimization)
        optimizer_A.step()
        optimizer_B.step()
        optimizer_C.step()
        optimizer_server.step()
        
        # ============================================
        # STEP 4: Update trigger using PGD (Paper's Eq. 3)
        # ============================================
        if should_update:
            # Update trigger using PGD with server gradients
            delta = backdoor_attack.update_trigger_pgd(
                partyA, XA, edge_index, 
                server_gradients=server_gradients,
                poison_nodes=poison_nodes
            )
            # Note: asr_poison will be computed during monitoring (STEP 5)
            backdoor_attack.best_delta_ever = backdoor_attack.delta.clone()
        
        # ============================================
        # STEP 5: Evaluation and monitoring
        # ============================================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Total Loss = {total_loss.item():.4f}, "
                  f"Clean Loss = {clean_loss.item():.4f}, "
                  f"Backdoor Loss = {backdoor_loss.item():.4f}, "
                  f"E = {E:.3f}")
            
            # Evaluate ASR on POISON NODES (where trigger was learned)
            partyA.eval()
            partyB.eval()
            partyC.eval()
            server.eval()
            with torch.no_grad():
                # Apply trigger to poison nodes (training targets) to verify backdoor works
                XA_triggered_poison = backdoor_attack.apply_trigger(XA, poison_nodes)
                
                HA_poison = partyA(XA_triggered_poison, edge_index)
                HB_poison = partyB(XB, edge_index)
                HC_poison = partyC(XC, edge_index)
                H_poison = torch.cat([HA_poison, HB_poison, HC_poison], dim=1)
                logits_poison = server(H_poison)
                
                target_labels_poison = torch.full((len(poison_nodes),), target_class, dtype=torch.long, device=device)
                asr_poison = attack_success_rate(logits_poison[poison_nodes], target_labels_poison, target_class)
                asr_history.append(asr_poison)
                
                # Track best ASR on poison nodes
                if asr_poison > backdoor_attack.best_asr_ever:
                    backdoor_attack.best_asr_ever = asr_poison
                    backdoor_attack.best_epoch_ever = epoch
                
                # Also evaluate on test set for reference
                test_nodes = test_mask.nonzero(as_tuple=False).view(-1)
                XA_triggered_test = backdoor_attack.apply_trigger(XA, test_nodes)
                
                HA_test = partyA(XA_triggered_test, edge_index)
                HB_test = partyB(XB, edge_index)
                HC_test = partyC(XC, edge_index)
                H_test = torch.cat([HA_test, HB_test, HC_test], dim=1)
                logits_test = server(H_test)
                
                target_labels_test = torch.full_like(y[test_mask], target_class)
                asr_test = attack_success_rate(logits_test[test_mask], target_labels_test, target_class)
                
                print(f"  ASR on POISON nodes (train): {asr_poison:.4f} | ASR on TEST nodes: {asr_test:.4f}")
                
            
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

        # Attack Evaluation - Apply trigger to test nodes
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


def run_vflip_defense_pipeline(
    partyA, partyB, partyC, server,
    XA, XB, XC, edge_index, y,
    train_mask, test_mask,
    threshold=-1.0,
    mae_epochs=600,
    device="cpu"
):
    """
    Apply VFLIP defense against trained backdoored models
    
    Returns dict with defense metrics and results
    """
    from defense.vflip.defense_pipeline import run_vflip_defense
    
    # Move data to device
    XA = XA.to(device)
    XB = XB.to(device)
    XC = XC.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Run VFLIP Defense
    vflip, defended_embeddings, defense_metrics = run_vflip_defense(
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
        threshold=threshold,
        mae_epochs=mae_epochs,
        device=device
    )
    
    print(f"\n🛡️ VFLIP Defense Applied Successfully")
    
    # Evaluate defense effectiveness
    partyA.eval()
    partyB.eval()
    partyC.eval()
    server.eval()
    
    with torch.no_grad():
        # Get anomaly detection results
        detected_mask, anomaly_scores = vflip.detect_anomalies(defended_embeddings)
        num_detected = detected_mask.sum().item()
        detection_rate = num_detected / len(detected_mask)
        
        # Compute defended accuracy
        test_outputs_defended = server(defended_embeddings)
        defended_predictions_all = test_outputs_defended.argmax(dim=1)
        test_labels = y[test_mask]
        defended_accuracy = (defended_predictions_all == test_labels).sum().item() / len(test_labels)
        
        # Compute original (attacked) ASR
        hA_all = partyA(XA, edge_index)
        hB_all = partyB(XB, edge_index)
        hC_all = partyC(XC, edge_index)
        all_embeddings_original = torch.cat([hA_all, hB_all, hC_all], dim=1)
        original_test_embeddings = all_embeddings_original[test_mask]
        
        original_outputs = server(original_test_embeddings)
        original_predictions = original_outputs.argmax(dim=1)
        num_triggered_original = (original_predictions == 0).sum().item()
        asr_original = num_triggered_original / len(original_test_embeddings)
        
        # Compute defended ASR
        defended_outputs = server(defended_embeddings)
        defended_predictions = defended_outputs.argmax(dim=1)
        
        # For detected anomalies, force non-backdoor prediction
        for i in range(len(detected_mask)):
            if detected_mask[i]:
                defended_outputs[i, 0] = defended_outputs[i].min() - 10
                defended_predictions[i] = defended_outputs[i].argmax()
        
        num_triggered_defended = (defended_predictions == 0).sum().item()
        asr_with_defense = num_triggered_defended / len(defended_embeddings)
    
    results = {
        'vflip': vflip,
        'defended_embeddings': defended_embeddings,
        'detection_rate': detection_rate,
        'defended_accuracy': defended_accuracy,
        'asr_original': asr_original,
        'asr_with_defense': asr_with_defense,
        'defense_metrics': defense_metrics
    }
    
    return results