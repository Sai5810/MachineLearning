import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(2)


class NeuralNet(nn.Module):

    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = layers
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

    def add(self, module):
        self.layers.append(module)
        self.stack = nn.Sequential(*self.layers)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:.3f}  [{current}/{size}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    class_acc = [[0, 0] for _ in enumerate(dataloader.dataset.classes)]
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for p, yi in zip(pred.argmax(1), y):
                if p == yi:
                    class_acc[yi][0] += 1
                class_acc[yi][1] += 1
    test_loss /= num_batches
    correct /= size
    for a, c in zip(class_acc, dataloader.dataset.classes):
        print(f"{c} Accuracy: {a[0] / a[1]:.3f}")
    print(class_acc)
    print(f"Accuracy: {(100 * correct)}%\nAverage Loss: {test_loss}\n")


def main():
    train_data = datasets.CIFAR10(root="data", train=True, transform=ToTensor())
    test_data = datasets.CIFAR10(root="data", transform=ToTensor())
    batch_size = 64
    train_dl = DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)
    model = NeuralNet([
        nn.Linear(32 * 32 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)])
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 3
    tic = time.perf_counter()
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train_loop(train_dl, model, loss_fn, optimizer)
        test_loop(test_dl, model, loss_fn)
    toc = time.perf_counter()
    print(f"Training epochs were finished in {toc - tic}s")


if __name__ == "__main__":
    main()
