import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    inputs = torch.tensor([[0.0], [1.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=True),
        torch.nn.Sigmoid()
    )
    summse = torch.nn.MSELoss(reduction="sum")
    opt = torch.optim.SGD(model[0].parameters(), lr=.1)
    print(f"Weight: {model[0].weight.data[0][0]}\nBias: {model[0].bias.data[0]}")
    line = [[], [], []]
    for i in range(10000):
        pred = model(inputs)
        model.zero_grad()
        loss = summse(inputs, pred)
        line[0].append(model[0].weight.data[0].item())
        line[1].append(model[0].bias.data.item())
        line[2].append(loss.data)
        loss.backward()
        opt.step()
    print(f"New Weight: {model[0].weight}\nNew Bias: {model[0].bias}")
    contour = [torch.linspace(-8, 8, 100), torch.linspace(-8, 8, 100), []]
    for w in contour[0]:
        t = []
        # model[0].weight.data = torch.tensor([[w]])
        model[0].weight.data.fill_(w)
        for b in contour[1]:
            # model[0].bias.data = b
            model[0].bias.data.fill_(b)
            pred = model(inputs)
            model.zero_grad()
            loss = summse(inputs, pred)
            t.append(loss.data)
        contour[2].append(t)
    ax = plt.figure().add_subplot(projection='3d')
    contour[2] = torch.tensor(contour[2])
    x, y = np.meshgrid(contour[0], contour[1])
    ax.plot_surface(x, y, contour[2], alpha=.5)
    ax.plot(line[0], line[1], line[2])
    ax.set_xlabel("Weight")
    ax.set_ylabel("Bias")
    ax.set_zlabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
