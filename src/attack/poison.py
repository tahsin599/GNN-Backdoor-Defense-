import torch
import random

def select_poison_nodes(y, target_class, poison_rate=0.05):
    candidates = (y != target_class).nonzero(as_tuple=True)[0]
    k = int(len(candidates) * poison_rate)
    return candidates[torch.randperm(len(candidates))[:k]]
