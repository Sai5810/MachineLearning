import torch
from tqdm import trange
from torch.nn.functional import mse_loss


def main():
    inputs = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    targets = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 8),
        torch.nn.Sigmoid()
    )
    print(f"Initial Parameters: {list(model[0].parameters())} \n {list(model[2].parameters())}")
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in trange(100000):
        model.zero_grad()
        pred = model(inputs)
        loss = mse_loss(pred, targets)
        loss.backward()
        opt.step()
    print(f"Final Parameters: {list(model[0].parameters())} \n {list(model[2].parameters())}")
    print(f"Loss: {loss.item()}")
    print(f"Inputs: {inputs}")
    print(f"Values after first layer {model[1](inputs)}")
    print(f"Predictions: {model(inputs)}")


if __name__ == "__main__":
    main()
