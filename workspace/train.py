"""Validation task for Adam: MNIST logistic regression (paper Section 6.1)."""
import torch, torch.nn as nn
from src.adam import Adam


def main(steps=500):
    torch.manual_seed(0)
    x = torch.randn(128, 784)
    y = torch.randint(0, 10, (128,))
    model = nn.Linear(784, 10)
    opt = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    lossf = nn.CrossEntropyLoss()
    for i in range(steps):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(f"step {i}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
