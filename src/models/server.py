import torch
import torch.nn as nn
import torch.nn.functional as F

class Server(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Server, self).__init__()
        self.num_classes = num_classes  # <--- store it
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
