import torch
import json

def vertical_split(features, split_ratio=0.5):
    num_features = features.shape[1]
    split = int(num_features * split_ratio)

    A = features[:, :split]
    B = features[:, split:]

    torch.save(A, "data/vertical_split/party_A_features.pt")
    torch.save(B, "data/vertical_split/party_B_features.pt")

    with open("data/vertical_split/feature_map.json", "w") as f:
        json.dump({
            "party_A": [0, split],
            "party_B": [split, num_features]
        }, f)

    return A, B
