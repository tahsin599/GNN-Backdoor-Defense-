import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv


class LocalGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.relu(x)
    




class LocalGAT(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=2):
        super().__init__()

        self.gat1 = GATConv(
            in_dim,
            hidden,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

