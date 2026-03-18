import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    """Fully Connected Network (FCN) used in MAE encoder and decoder."""
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
    Masked Autoencoder (MAE) for VFLIP defense.
    
    The MAE is trained on clean embeddings and learns to:
    1. Identify anomalies via reconstruction error
    2. Purify backdoor-triggered embeddings by reconstruction
    
    Paper: VFLIP: A Backdoor Defense for Vertical Federated Learning
    """
    
    def __init__(self, embedding_dim, hidden_dim=128, mask_ratio=0.15, dropout=0.1):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension of FCN encoder/decoder
            mask_ratio: Ratio of embeddings to mask during training
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        
        # Encoder: Maps embeddings to latent representation
        self.encoder = FCN(embedding_dim, hidden_dim, hidden_dim // 2, dropout)
        
        # Decoder: Reconstructs embeddings from latent representation
        self.decoder = FCN(hidden_dim // 2, hidden_dim, embedding_dim, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass through MAE.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            mask: Binary mask indicating which elements to mask (optional)
            
        Returns:
            reconstructed: Reconstructed embeddings
            latent: Latent representation
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def compute_reconstruction_error(self, x):
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            
        Returns:
            error: Reconstruction error per sample (batch_size,)
        """
        reconstructed, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error
    
    def compute_anomaly_score(self, x):
        """
        Compute anomaly score for backdoor detection.
        
        Based on reconstruction error with normalization.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            
        Returns:
            anomaly_score: Normalized anomaly score per sample
        """
        error = self.compute_reconstruction_error(x)
        # Normalize using z-score normalization
        mean_error = error.mean()
        std_error = error.std() + 1e-8
        anomaly_score = (error - mean_error) / std_error
        return anomaly_score
    
    def purify_embeddings(self, x, iterations=1):
        """
        Purify embeddings by iterative reconstruction.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            iterations: Number of purification iterations
            
        Returns:
            purified: Purified embeddings
        """
        with torch.no_grad():
            purified = x.clone()
            for _ in range(iterations):
                reconstructed, _ = self.forward(purified)
                # Blend original and reconstructed for gradual purification
                purified = 0.7 * reconstructed + 0.3 * purified
        return purified
    
    def train_on_clean_embeddings(self, clean_embeddings, epochs=20, lr=0.01, device='cpu'):
        """
        Train MAE on clean embeddings.
        
        Args:
            clean_embeddings: Clean embeddings for training (n_samples, embedding_dim)
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to use for training
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        clean_embeddings = clean_embeddings.to(device)
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            reconstructed, _ = self.forward(clean_embeddings)
            loss = criterion(reconstructed, clean_embeddings)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  MAE Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.eval()
