import torch


def main():
    inputs = torch.tensor([[0.0], [1.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=True),
        torch.nn.Sigmoid()
    )
    summse = torch.nn.MSELoss(reduction="sum")
    print(f"Weight: {model[0].weight.data}\nBias: {model[0].bias.data}")
    opt = torch.optim.SGD(model[0].parameters(), lr=.1)
    for i in range(10000):
        pred = model(inputs)
        model.zero_grad()
        loss = summse(inputs, pred)
        print(loss)
        loss.backward()
        opt.step()
    print(f"New Weight: {model[0].weight}\nNew Bias: {model[0].bias}")


if __name__ == "__main__":
    main()
