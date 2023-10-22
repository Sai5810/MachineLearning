import torch


def main():
    inputs = torch.tensor([[0.0], [1.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=True),
        torch.nn.Sigmoid()
    )
    vals = [(20.0, -40.0), (-20.0, 40.0), (0.0, 0.0), (1.0, -1.0), (-1.0, 1.0)]
    with torch.no_grad():
        for b, w in vals:
            print(f"Weight: {w}, Bias: {b}")
            model[0].weight.fill_(w)
            model[0].bias.fill_(b)
            pred = model(inputs)
            for i, p in inputs, pred:
                print(f"Input: {i[0]}, Pred: {p[0]}")


if __name__ == "__main__":
    main()
