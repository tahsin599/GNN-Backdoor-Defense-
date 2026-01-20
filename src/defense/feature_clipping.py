import torch

def feature_clipping(X, min_val=-3.0, max_val=3.0):
    return torch.clamp(X, min=min_val, max=max_val)
