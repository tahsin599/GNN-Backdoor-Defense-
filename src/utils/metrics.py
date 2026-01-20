import torch


def accuracy(logits, labels):
    """
    Standard classification accuracy
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def attack_success_rate(logits, labels, target_class):
    """
    ASR: fraction of samples predicted as target_class
    """
    preds = logits.argmax(dim=1)
    return (preds == target_class).float().mean().item()
