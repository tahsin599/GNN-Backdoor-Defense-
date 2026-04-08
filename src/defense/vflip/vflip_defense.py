import torch
import numpy as np
from .masked_autoencoder import MaskedAutoencoder


class VFLIPDefense:
    def __init__(self, party_splits, threshold_percentile=95.0,
                 mae_hidden_dim=256, mae_dropout=0.1, device='cpu',
                 use_weighted_voting=True):  # NEW: enable/disable weighted voting
        self.party_splits = party_splits
        self.num_parties = len(party_splits)
        self.embedding_dim = party_splits[-1][1]
        self.threshold_percentile = threshold_percentile
        self.device = device
        self.use_weighted_voting = use_weighted_voting  # NEW

        self.mae = MaskedAutoencoder(
            embedding_dim=self.embedding_dim,
            party_splits=party_splits,
            hidden_dim=mae_hidden_dim,
            dropout=mae_dropout
        ).to(device)

        self.thresholds = None
        self.party_weights = None  # NEW: reliability weights for each party
        self.party_weighted_threshold = None  # NEW: adjusted threshold for weighted voting

    def train_mae_on_clean_embeddings(self, clean_embeddings,
                                      epochs=20, lr1=0.01, lr2=0.01):
        print("\nTraining MAE on clean embeddings...")
        self.mae.train_on_clean_embeddings(
            clean_embeddings, epochs, lr1, lr2, self.device
        )

        print("Computing per‑party anomaly thresholds...")
        self.thresholds, self.party_weights = self._compute_thresholds_and_weights(clean_embeddings)
        
        for i, t in enumerate(self.thresholds):
            print(f"  Party {i} threshold t_{i} = {t:.6f}")
        
        if self.use_weighted_voting and self.party_weights is not None:
            print(f"  Party reliability weights: {self.party_weights.cpu().numpy()}")
            # Weighted threshold: sum of weights / 2 (instead of N/2)
            self.party_weighted_threshold = self.party_weights.sum() / 2
            print(f"  Weighted voting threshold: {self.party_weighted_threshold:.4f}")

    def _compute_thresholds_and_weights(self, clean_embeddings):
        """
        Compute thresholds AND party reliability weights.
        Returns:
            thresholds: per-party anomaly thresholds
            party_weights: normalized reliability weights (higher = more reliable)
        """
        self.mae.eval()
        n = self.num_parties
        all_scores = {i: [] for i in range(n)}
        batch_size = 256
        num_samples = clean_embeddings.size(0)

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = clean_embeddings[start:end].to(self.device)

                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        mask_input = torch.zeros(batch.size(0), n, device=self.device)
                        mask_input[:, j] = 1.0
                        mask_target = torch.zeros(batch.size(0), n, device=self.device)
                        mask_target[:, i] = 1.0

                        loss = self.mae.compute_reconstruction_error_on_party(
                            batch, mask_input, mask_target
                        )
                        all_scores[i].extend(loss.cpu().numpy())

        # Compute thresholds (percentile-based)
        thresholds = []
        for i in range(n):
            scores_i = np.array(all_scores[i])
            t = np.percentile(scores_i, self.threshold_percentile)
            thresholds.append(t)
        
        # NEW: Compute party reliability weights
        # A party is reliable if others can reconstruct it well (low average error)
        party_weights = []
        for i in range(n):
            avg_error = np.mean(all_scores[i])
            # Lower error = higher weight
            weight = 1.0 / (avg_error + 1e-8)
            party_weights.append(weight)
        
        # Normalize weights so they sum to number of parties (for fair comparison)
        party_weights = np.array(party_weights)
        party_weights = party_weights / party_weights.sum() * n
        party_weights = torch.tensor(party_weights, dtype=torch.float, device=self.device)
        
        return thresholds, party_weights

    def identify_malicious_parties(self, h):
        batch_size = h.size(0)
        n = self.num_parties
        h = h.to(self.device)
        self.mae.eval()

        if self.use_weighted_voting and self.party_weights is not None:
            # WEIGHTED VOTING (Improved)
            # Each vote is weighted by the reliability of the context party
            weighted_votes = torch.zeros(batch_size, n, dtype=torch.float, device=self.device)

            with torch.no_grad():
                for i in range(n):  # target party
                    for j in range(n):  # context party
                        if i == j:
                            continue
                        mask_input = torch.zeros(batch_size, n, device=self.device)
                        mask_input[:, j] = 1.0
                        mask_target = torch.zeros(batch_size, n, device=self.device)
                        mask_target[:, i] = 1.0

                        s_j_to_i = self.mae.compute_reconstruction_error_on_party(
                            h, mask_input, mask_target
                        )
                        is_anomaly = s_j_to_i > self.thresholds[i]
                        # Weight by reliability of the context party (j)
                        weighted_votes[is_anomaly, i] += self.party_weights[j]

            malicious_mask = weighted_votes > self.party_weighted_threshold
            
        else:
            # ORIGINAL VOTING (Majority)
            votes = torch.zeros(batch_size, n, dtype=torch.int, device=self.device)

            with torch.no_grad():
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        mask_input = torch.zeros(batch_size, n, device=self.device)
                        mask_input[:, j] = 1.0
                        mask_target = torch.zeros(batch_size, n, device=self.device)
                        mask_target[:, i] = 1.0

                        s_j_to_i = self.mae.compute_reconstruction_error_on_party(
                            h, mask_input, mask_target
                        )
                        is_anomaly = s_j_to_i > self.thresholds[i]
                        votes[is_anomaly, i] += 1

            malicious_mask = votes > (n / 2)

        return malicious_mask

    def purify_embeddings(self, h, malicious_mask):
        batch_size = h.size(0)
        h = h.to(self.device)
        self.mae.eval()

        mask_input = (~malicious_mask).float()
        h_purified = self.mae.forward(h, mask_input)
        return h_purified

    def defend_on_batch(self, h):
        malicious_mask = self.identify_malicious_parties(h)
        defended = self.purify_embeddings(h, malicious_mask)
        return defended, malicious_mask