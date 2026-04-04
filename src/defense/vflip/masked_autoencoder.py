import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCN(nn.Module):
    """Fully Connected Network used in MAE encoder and decoder."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for VFLIP.
    Trained with two strategies:
      - "N-1 to 1": predict one party's embedding from all others.
      - "1 to 1":   predict one party's embedding from another single party.
    """

    def __init__(self, embedding_dim, party_splits, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.party_splits = party_splits  # list of (start, end) indices
        self.num_parties = len(party_splits)

        self.encoder = FCN(embedding_dim, hidden_dim, hidden_dim // 2, dropout)
        self.decoder = FCN(hidden_dim // 2, hidden_dim, embedding_dim, dropout)

    def apply_block_mask(self, x, mask):
        """
        mask: (batch, num_parties) – 1 means keep, 0 means zero out that party's block.
        """
        x_masked = x.clone()
        for p, (start, end) in enumerate(self.party_splits):
            party_mask = mask[:, p].float().unsqueeze(1)  # (batch, 1)
            x_masked[:, start:end] = x_masked[:, start:end] * party_mask
        return x_masked

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size(0), self.num_parties, device=x.device)
        x_masked = self.apply_block_mask(x, mask)
        latent = self.encoder(x_masked)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error_on_party(self, x, mask_input, mask_target):
        """
        mask_input: (batch, num_parties) – which parties are given to MAE.
        mask_target: (batch, num_parties) – which parties to compute loss on.
        Returns MSE per sample over the target block only.
        """
        reconstructed = self.forward(x, mask_input)
        # Build expanded target mask for embedding dimension
        target_mask_expanded = torch.zeros_like(x)
        for p, (start, end) in enumerate(self.party_splits):
            target_mask_expanded[:, start:end] = mask_target[:, p].float().unsqueeze(1)
        diff = (x - reconstructed) * target_mask_expanded
        loss_per_sample = (diff ** 2).sum(dim=1) / (target_mask_expanded.sum(dim=1).clamp(min=1))
        return loss_per_sample

    def train_on_clean_embeddings(self, clean_embeddings, epochs=20,
                                  lr1=0.01, lr2=0.01, device='cpu'):
        self.to(device)
        clean_embeddings = clean_embeddings.to(device)
        n = self.num_parties

        optimizer1 = torch.optim.Adam(self.parameters(), lr=lr1)
        optimizer2 = torch.optim.Adam(self.parameters(), lr=lr2)

        for epoch in range(epochs):
            perm = torch.randperm(clean_embeddings.size(0))
            embeddings_shuffled = clean_embeddings[perm]
            batch = embeddings_shuffled  # full batch for simplicity

            # ---- N-1 to 1 ----
            optimizer1.zero_grad()
            loss_n1 = 0.0
            for h in batch:
                i = np.random.randint(0, n)
                mask_input = torch.ones(n, device=device)
                mask_input[i] = 0.0
                mask_input = mask_input.unsqueeze(0)
                mask_target = torch.zeros(n, device=device)
                mask_target[i] = 1.0
                mask_target = mask_target.unsqueeze(0)
                loss_i = self.compute_reconstruction_error_on_party(
                    h.unsqueeze(0), mask_input, mask_target
                )
                loss_n1 += loss_i
            loss_n1 /= len(batch)
            loss_n1.backward()
            optimizer1.step()

            # ---- 1 to 1 ----
            optimizer2.zero_grad()
            loss_11 = 0.0
            for h in batch:
                parties = np.random.choice(n, size=2, replace=False)
                i, j = parties[0], parties[1]
                mask_input = torch.zeros(n, device=device)
                mask_input[j] = 1.0
                mask_input = mask_input.unsqueeze(0)
                mask_target = torch.zeros(n, device=device)
                mask_target[i] = 1.0
                mask_target = mask_target.unsqueeze(0)
                loss_ij = self.compute_reconstruction_error_on_party(
                    h.unsqueeze(0), mask_input, mask_target
                )
                loss_11 += loss_ij
            loss_11 /= len(batch)
            loss_11.backward()
            optimizer2.step()

            if (epoch + 1) % 5 == 0:
                print(f"MAE Epoch {epoch+1}/{epochs} | N-1→1 loss: {loss_n1.item():.6f} | 1→1 loss: {loss_11.item():.6f}")

        self.eval()