import torch
import numpy

def main():
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    summse = torch.nn.MSELoss(reduction="sum")
    print(f"Initial Parameters: {list(model[0].parameters())}")
    opt = torch.optim.SGD(model[0].parameters(), lr=.1)
    for i in range(10000):
        pred = model(inputs)
        model.zero_grad()
        loss = summse(pred, targets)
        # print(loss.item())
        loss.backward()
        opt.step()
    print(f"Final Parameters: {list(model[0].parameters())}")
    print(f"Loss: {loss.item()}")
    print(f"Inputs: {inputs}")
    print(f"Predictions: {model(inputs)}")


if __name__ == "__main__":
    main()
