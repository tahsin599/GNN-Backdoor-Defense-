import torch
import torch.nn.functional as F

def fine_tune_server(server, h, y, mask, epochs=20):
    optimizer = torch.optim.Adam(server.parameters(), lr=0.005)
    server.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = server(h)
        loss = F.cross_entropy(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
