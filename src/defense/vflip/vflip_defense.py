import torch
import numpy as np
from .masked_autoencoder import MaskedAutoencoder


class VFLIPDefense:
    def __init__(self, party_splits, threshold_percentile=95.0,
                 mae_hidden_dim=256, mae_dropout=0.1, device='cpu'):
        self.party_splits = party_splits
        self.num_parties = len(party_splits)
        self.embedding_dim = party_splits[-1][1]
        self.threshold_percentile = threshold_percentile
        self.device = device

        self.mae = MaskedAutoencoder(
            embedding_dim=self.embedding_dim,
            party_splits=party_splits,
            hidden_dim=mae_hidden_dim,
            dropout=mae_dropout
        ).to(device)

        self.thresholds = None

    def train_mae_on_clean_embeddings(self, clean_embeddings,
                                      epochs=20, lr1=0.01, lr2=0.01):
        print("\nTraining MAE on clean embeddings...")
        self.mae.train_on_clean_embeddings(
            clean_embeddings, epochs, lr1, lr2, self.device
        )

        print("Computing per‑party anomaly thresholds...")
        self.thresholds = self._compute_thresholds(clean_embeddings)
        for i, t in enumerate(self.thresholds):
            print(f"  Party {i} threshold t_{i} = {t:.6f}")

    def _compute_thresholds(self, clean_embeddings):
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

        thresholds = []
        for i in range(n):
            scores_i = np.array(all_scores[i])
            t = np.percentile(scores_i, self.threshold_percentile)
            thresholds.append(t)
        return thresholds

    def identify_malicious_parties(self, h):
        batch_size = h.size(0)
        n = self.num_parties
        h = h.to(self.device)
        self.mae.eval()

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

        malicious_mask = votes > (n // 2)
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