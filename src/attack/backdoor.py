import torch
import torch.nn.functional as F
from torch.autograd import grad
import random

class VFGNNAttack:
    def __init__(
        self,
        epsilon=1.0,
        alpha=0.1,
        num_hops=2,
        target_class=0,
        poison_ratio=0.05,
        similarity_threshold=0.8,  # τ in paper
        device="cpu"
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_hops = num_hops
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.similarity_threshold = similarity_threshold  # τ threshold
        self.device = device
        
        # Initialize trigger
        self.delta = None
        self.poison_nodes = None  # VP in paper
        self.multi_hop_neighbors = None
        
        # Track best trigger ever found
        self.best_delta_ever = None
        self.best_asr_ever = 0.0
        self.best_epoch_ever = -1
        
        # Backdoor Retention tracking
        self.prev_delta = None  # For retention
        self.prev_model_state = None  # For retention
        self.similarity_history = []  # Track E over epochs
        
    def compute_multi_hop_neighbors(self, edge_index, num_nodes):
        """Precompute multi-hop neighbors"""
        adj = torch.zeros((num_nodes, num_nodes), device=self.device, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        adj_powers = [torch.eye(num_nodes, device=self.device, dtype=torch.float32)]
        adj_powers.append(adj.clone())
        
        for hop in range(2, self.num_hops + 1):
            if adj_powers[-1].dtype != torch.float32:
                prev_power = adj_powers[-1].float()
            else:
                prev_power = adj_powers[-1]
            
            if adj.dtype != torch.float32:
                adj_float = adj.float()
            else:
                adj_float = adj
            
            next_power = torch.mm(prev_power, adj_float)
            adj_powers.append(next_power)
        
        multi_hop_neighbors = []
        for hop in range(self.num_hops + 1):
            neighbors = []
            for node in range(num_nodes):
                if hop == 0:
                    neighbors.append(torch.tensor([node], device=self.device, dtype=torch.long))
                else:
                    node_neighbors = (adj_powers[hop][node] > 0).nonzero(as_tuple=False).view(-1)
                    
                    if hop > 0:
                        for h in range(hop):
                            prev_neighbors = (adj_powers[h][node] > 0).nonzero(as_tuple=False).view(-1)
                            if len(prev_neighbors) > 0 and len(node_neighbors) > 0:
                                mask = ~torch.isin(node_neighbors, prev_neighbors)
                                node_neighbors = node_neighbors[mask]
                    
                    neighbors.append(node_neighbors)
            multi_hop_neighbors.append(neighbors)
        
        return multi_hop_neighbors
    
    def project_to_epsilon_ball(self, delta_tensor):
        """Π_ϵ from paper's Eq. (3)"""
        norm = torch.norm(delta_tensor, p=2)
        if norm > self.epsilon:
            delta_tensor = delta_tensor * self.epsilon / norm
        return delta_tensor
    
    def compute_backdoor_similarity(self, model, XA, edge_index):
        """
        Compute backdoor effectiveness E using Eq. (6) from paper:
        E = (1/n²) Σ_i Σ_j (H_i·H_j) / (∥H_i∥·∥H_j∥)
        """
        if self.poison_nodes is None or len(self.poison_nodes) == 0:
            return 0.0
        
        # Apply trigger to poisoned nodes
        XA_triggered = self.apply_trigger(XA, self.poison_nodes)
        
        # Get embeddings H for poisoned nodes (from adversary's bottom model)
        with torch.no_grad():
            H = model(XA_triggered, edge_index)
            H_poisoned = H[self.poison_nodes]  # Get embeddings for poisoned nodes
        
        n = H_poisoned.shape[0]
        if n == 0:
            return 0.0
        
        # Normalize embeddings
        H_norm = F.normalize(H_poisoned, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(H_norm, H_norm.T)
        
        # Compute E as average similarity (excluding diagonal)
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        E = similarity_matrix[mask].mean().item()
        
        return E
    
    def should_update_backdoor(self, E, epoch):
        """
        Backdoor Retention decision logic from paper:
        - If E ≥ threshold: update model and trigger
        - Otherwise: retain previous model and trigger
        """
        self.similarity_history.append({'epoch': epoch, 'E': E})
        
        if E >= self.similarity_threshold:
            return True
        else:
            return False
    
    def update_trigger_pgd(self, model, XA, edge_index, server_gradients=None, poison_nodes=None):
        """
        Paper's Eq. (3): δ^{t+1} = Π_ϵ(δ^t - α·sgn(∇δL(F(a(Gp, δ^t); Θ*), τ)))
        
        Follows Algorithm 1 line 13-14:
        - computes ∇δL with {∂L/∂Hi * ∂Hi/∂δ} for vi ∈ VP
        - updates δ with Eq. (3)
        """
        if self.delta is None or poison_nodes is None:
            return self.delta
        
        if server_gradients is None:
            # Fallback if server gradients not available
            return self._update_trigger_simple(model, XA, edge_index, poison_nodes)
        
        # Create poisoned features for gradient computation
        XA_poisoned = XA.clone().detach()
        with torch.no_grad():
            for node in poison_nodes:
                for hop in range(self.num_hops + 1):
                    neighbors = self.multi_hop_neighbors[hop][node]
                    if len(neighbors) > 0:
                        XA_poisoned[neighbors] = XA_poisoned[neighbors] + self.delta[neighbors]
        
        # Need gradients w.r.t XA_poisoned
        XA_poisoned = XA_poisoned.requires_grad_(True)
        
        # Forward pass to get H (embeddings from adversary's model)
        H = model(XA_poisoned, edge_index)
        
        # Get server gradients for poisoned nodes
        # These are ∂L/∂H from the active party (Algorithm 1 line 11)
        server_grads_for_poisoned = server_gradients[poison_nodes]
        
        # Compute loss for gradient computation
        # We need: ∇δL = ∂L/∂δ = (∂L/∂H) * (∂H/∂δ)
        # Multiply H by server gradients and sum to get scalar loss
        loss = (H[poison_nodes] * server_grads_for_poisoned).sum()
        
        # Compute gradient w.r.t. XA_poisoned (which gives ∂H/∂δ)
        gradients = torch.autograd.grad(loss, XA_poisoned, retain_graph=False)[0]
        
        # Create mask for poison nodes and their neighbors
        update_mask = torch.zeros_like(XA, device=self.device)
        for node in poison_nodes:
            for hop in range(self.num_hops + 1):
                neighbors = self.multi_hop_neighbors[hop][node]
                if len(neighbors) > 0:
                    update_mask[neighbors] = 1
        
        # Apply mask
        masked_gradients = gradients * update_mask
        
        # Apply PGD update: δ^{t+1} = Π_ϵ(δ^t - α·sgn(∇δL))
        with torch.no_grad():
            update = self.alpha * torch.sign(masked_gradients)
            self.delta = self.delta - update
            self.delta = self.project_to_epsilon_ball(self.delta)
        
        return self.delta
    
    def _update_trigger_simple(self, model, XA, edge_index, poison_nodes):
        """Simplified trigger update when server gradients are not available"""
        if self.delta is None or poison_nodes is None:
            return self.delta
        
        # Create poisoned features
        XA_poisoned = XA.clone().detach()
        with torch.no_grad():
            for node in poison_nodes:
                for hop in range(self.num_hops + 1):
                    neighbors = self.multi_hop_neighbors[hop][node]
                    if len(neighbors) > 0:
                        XA_poisoned[neighbors] = XA_poisoned[neighbors] + self.delta[neighbors]
        
        XA_poisoned = XA_poisoned.requires_grad_(True)
        
        # Forward pass
        H = model(XA_poisoned, edge_index)
        H_poisoned = H[poison_nodes]
        
        # Create target direction (simplified backdoor objective)
        target_direction = torch.zeros_like(H_poisoned)
        target_direction[:, 0] = 1.0  # First dimension indicates target class
        
        loss = F.mse_loss(H_poisoned, target_direction)
        
        # Compute gradients
        gradients = torch.autograd.grad(loss, XA_poisoned, retain_graph=False)[0]
        
        # Create mask
        update_mask = torch.zeros_like(XA, device=self.device)
        for node in poison_nodes:
            for hop in range(self.num_hops + 1):
                neighbors = self.multi_hop_neighbors[hop][node]
                if len(neighbors) > 0:
                    update_mask[neighbors] = 1
        
        masked_gradients = gradients * update_mask
        
        # Apply PGD update
        with torch.no_grad():
            update = self.alpha * torch.sign(masked_gradients)
            self.delta = self.delta - update
            self.delta = self.project_to_epsilon_ball(self.delta)
        
        return self.delta
    
    def initialize_attack(self, XA, edge_index, train_mask, y=None):
        """
        Initialize attack with poison ratio
        Selects VP (target class nodes available to adversary)
        """
        train_idx = train_mask.nonzero(as_tuple=False).view(-1)
        num_poison = max(4, int(len(train_idx) * self.poison_ratio))
        
        # Select target class nodes (VP in paper - line 2 of Algorithm 1)
        if y is not None:
            target_in_train = train_idx[y[train_idx] == self.target_class]
            if len(target_in_train) >= num_poison:
                self.poison_nodes = target_in_train[:num_poison]
            else:
                remaining = train_idx[~torch.isin(train_idx, target_in_train)]
                needed = num_poison - len(target_in_train)
                supplement = remaining[torch.randperm(len(remaining))[:needed]]
                self.poison_nodes = torch.cat([target_in_train, supplement])
        else:
            self.poison_nodes = train_idx[torch.randperm(len(train_idx))[:num_poison]]
        
        print(f"Selected {len(self.poison_nodes)} nodes for poisoning (VP)")
        print(f"({self.poison_ratio*100:.1f}% of training, min 4)")
        
        # Initialize δ = 0 as per paper Algorithm 1 line 1
        self.delta = torch.zeros_like(XA, device=self.device)
        self.prev_delta = self.delta.clone()  # For BR
        
        # Precompute multi-hop neighbors for Eq. (4)
        num_nodes = XA.shape[0]
        self.multi_hop_neighbors = self.compute_multi_hop_neighbors(edge_index, num_nodes)
        
        return self.poison_nodes
    
    def apply_trigger(self, XA, nodes=None):
        """
        Apply trigger to nodes according to Eq. (4):
        a(Gp, δ) = (xp+δ⁰, X¹⁻ʰᵒᵖ+δ¹, ···, Xᴹ⁻ʰᵒᵖ+δᴹ)
        """
        if self.delta is None:
            return XA.clone()
        
        XA_poisoned = XA.clone()
        if nodes is None:
            nodes = self.poison_nodes if self.poison_nodes is not None else []
        
        with torch.no_grad():
            for node in nodes:
                for hop in range(self.num_hops + 1):
                    neighbors = self.multi_hop_neighbors[hop][node]
                    if len(neighbors) > 0:
                        # Apply same δ to all hops (paper uses δ^m but same in implementation)
                        XA_poisoned[neighbors] = XA_poisoned[neighbors] + self.delta[neighbors]
        
        return XA_poisoned
    
    def get_best_trigger_info(self):
        if self.best_delta_ever is None:
            return None
        
        return {
            'delta': self.best_delta_ever,
            'asr': self.best_asr_ever,
            'epoch': self.best_epoch_ever,
            'norm': torch.norm(self.best_delta_ever).item(),
            'similarity_history': self.similarity_history
        }