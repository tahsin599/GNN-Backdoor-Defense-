from torch_geometric.datasets import Planetoid
import torch
import os

def load_cora(root="data"):
   
    os.makedirs(root, exist_ok=True)
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    # Save as .pt files for future use
    torch.save(data.edge_index, os.path.join(root, "edges.pt"))
    torch.save(data.x, os.path.join(root, "features.pt"))
    torch.save(data.y, os.path.join(root, "labels.pt"))

    print("Cora dataset loaded via torch_geometric and saved as .pt files.")
    return data


