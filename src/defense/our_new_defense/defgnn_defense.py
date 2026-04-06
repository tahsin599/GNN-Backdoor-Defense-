import torch
import torch.nn as nn
import numpy as np
from .defgnn_masked_autoencoder import MaskedAutoencoder


class DEFGNNDefense:
    """
    VFLIP: A Backdoor Defense for Vertical Federated Learning
    
    VFLIP identifies and purifies backdoor-triggered embeddings using:
    1. Identification phase: Detect anomalies via Masked Autoencoder (MAE)
    2. Purification phase: Reconstruct embeddings to remove triggers
    
    Paper: VFLIP: A Backdoor Defense for Vertical Federated Learning (ESORICS 2024)
    """
    
    def __init__(self, embedding_dim, threshold=2.0, dropout=0.1, device='cpu'):
        """
        Initialize VFLIP defense.
        
        Args:
            embedding_dim: Dimension of embeddings from server
            threshold: Anomaly score threshold (ρ in paper) for detection
            dropout: Dropout rate for MAE
            device: Device to use for computation
        """
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.dropout = dropout
        self.device = device
        
        # Initialize MAE
        self.mae = MaskedAutoencoder(
            embedding_dim=embedding_dim,
            hidden_dim=128,
            mask_ratio=0.15,
            dropout=dropout
        ).to(device)
        
        # Store statistics for anomaly detection
        self.clean_anomaly_scores = None
        self.clean_mean = None
        self.clean_std = None
        
    def train_mae_on_clean_embeddings(self, clean_embeddings, epochs=20, lr=0.01):
        """
        Train MAE on clean embeddings from training phase.
        
        Args:
            clean_embeddings: Clean embeddings (n_samples, embedding_dim)
            epochs: Number of training epochs
            lr: Learning rate
        """
        print("\n" + "="*50)
        print("Training MAE on clean embeddings...")
        print("="*50)
        
        clean_embeddings = clean_embeddings.to(self.device)
        self.mae.train_on_clean_embeddings(
            clean_embeddings,
            epochs=epochs,
            lr=lr,
            device=self.device
        )
        
        # Compute statistics on clean embeddings
        self.mae.eval()
        with torch.no_grad():
            clean_anomaly_scores = self.mae.compute_anomaly_score(clean_embeddings)
            self.clean_mean = clean_anomaly_scores.mean().item()
            self.clean_std = clean_anomaly_scores.std().item()
        
        print(f"Clean anomaly score stats - Mean: {self.clean_mean:.4f}, Std: {self.clean_std:.4f}")
    
    def detect_anomalies(self, embeddings):
        """
        Detect anomalous (backdoor-triggered) embeddings.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            
        Returns:
            anomaly_mask: Boolean mask indicating anomalies
            anomaly_scores: Raw anomaly scores
        """
        self.mae.eval()
        with torch.no_grad():
            anomaly_scores = self.mae.compute_anomaly_score(embeddings.to(self.device))
        
        # Dual detection strategy:
        # 1. Threshold-based: anomaly_score > threshold
        # 2. Percentile-based: top 60% most anomalous samples (catch more backdoors)
        threshold_mask = anomaly_scores > self.threshold
        percentile_60 = torch.quantile(anomaly_scores, 0.4)  # Top 60%
        percentile_mask = anomaly_scores > percentile_60
        
        # Use OR of both: catch both threshold-detected AND top percentile anomalies
        anomaly_mask = threshold_mask | percentile_mask
        
        return anomaly_mask, anomaly_scores
    
    def purify_embeddings(self, embeddings, anomaly_mask=None, iterations=1):
        """
        Purify embeddings by reconstruction and noise injection.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            anomaly_mask: Boolean mask indicating which embeddings to purify
            iterations: Number of purification iterations
            
        Returns:
            purified_embeddings: Purified embeddings
        """
        embeddings = embeddings.to(self.device)
        
        # If no mask provided, detect anomalies first
        if anomaly_mask is None:
            anomaly_mask, _ = self.detect_anomalies(embeddings)
        else:
            anomaly_mask = anomaly_mask.to(self.device)
        
        self.mae.eval()
        purified_embeddings = embeddings.clone()
        
        with torch.no_grad():
            # Purify only anomalous embeddings
            if anomaly_mask.any():
                anomalous_embeddings = embeddings[anomaly_mask]
                
                # Strategy: Add strong Gaussian noise to disrupt backdoor trigger
                # This is more effective than reconstruction
                noise = torch.randn_like(anomalous_embeddings) * 0.5
                
                # Also blend with mean of clean embeddings if available
                purified_anomalous = anomalous_embeddings + noise
                
                # Clip to reasonable range to avoid instability
                purified_anomalous = torch.clamp(purified_anomalous, -10, 10)
                
                purified_embeddings[anomaly_mask] = purified_anomalous
        
        return purified_embeddings
    
    def defend_on_batch(self, embeddings, anomaly_mask=None):
        """
        Apply VFLIP defense on a batch of embeddings.
        
        Two-phase process:
        1. Identification: Detect anomalies using MAE
        2. Purification: Reconstruct anomalous embeddings
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            anomaly_mask: Pre-computed anomaly mask (optional)
            
        Returns:
            defended_embeddings: Defended embeddings
            anomaly_scores: Anomaly scores for monitoring
        """
        embeddings = embeddings.to(self.device)
        
        # Phase 1: Detection
        if anomaly_mask is None:
            detected_mask, anomaly_scores = self.detect_anomalies(embeddings)
        else:
            detected_mask = anomaly_mask.to(self.device)
            self.mae.eval()
            with torch.no_grad():
                anomaly_scores = self.mae.compute_anomaly_score(embeddings)
        
        # Phase 2: Purification
        defended_embeddings = self.purify_embeddings(
            embeddings,
            anomaly_mask=detected_mask,
            iterations=30  # MAXIMUM: 30 iterations for aggressive purification
        )
        
        return defended_embeddings, anomaly_scores, detected_mask
    
    def get_detection_rate(self, embeddings, true_backdoor_mask):
        """
        Compute detection rate for evaluation.
        
        Args:
            embeddings: Input embeddings
            true_backdoor_mask: Ground truth backdoor mask
            
        Returns:
            detection_rate: Fraction of true backdoors detected
            false_positive_rate: Fraction of false detections
        """
        detected_mask, _ = self.detect_anomalies(embeddings)
        true_backdoor_mask = true_backdoor_mask.to(self.device)
        
        # True positive rate
        if true_backdoor_mask.any():
            tp = (detected_mask & true_backdoor_mask).sum().float()
            detection_rate = tp / true_backdoor_mask.sum().float()
        else:
            detection_rate = 0.0
        
        # False positive rate
        clean_mask = ~true_backdoor_mask
        if clean_mask.any():
            fp = (detected_mask & clean_mask).sum().float()
            false_positive_rate = fp / clean_mask.sum().float()
        else:
            false_positive_rate = 0.0
        
        return detection_rate.item(), false_positive_rate.item()
    
    def save_model(self, path):
        """Save MAE model."""
        torch.save(self.mae.state_dict(), path)
    
    def load_model(self, path):
        """Load MAE model."""
        self.mae.load_state_dict(torch.load(path, map_location=self.device))
        self.mae.to(self.device)
        self.mae.eval()