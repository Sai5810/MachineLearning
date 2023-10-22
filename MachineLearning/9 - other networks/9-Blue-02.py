import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torch.nn.functional as func
torch.manual_seed(2)


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch + 1) * len(X) % 10000 == 0:
            print(f"Loss: {loss.item():.3f}  [{(batch + 1) * len(X)}/{size}]")


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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root="data", train=True, transform=transform)
    test_data = datasets.CIFAR10(root="data", transform=transform)
    batch_size = 4
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_data, batch_size=batch_size, num_workers=2)
    model = NeuralNet()
    try:
        model.load_state_dict(torch.load("C:/Users/saiva/PycharmProjects/9 - other networks/CNN.pt"))
        model.eval()
    except FileNotFoundError:
        pass
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
    epochs = 3
    tic = time.perf_counter()
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train_loop(train_dl, model, loss_fn, optimizer)
        test_loop(test_dl, model, loss_fn)
    toc = time.perf_counter()
    print(f"Training epochs were finished in {toc - tic}s")
    torch.save(model.state_dict(), "C:/Users/saiva/PycharmProjects/9 - other networks/CNN.pt")


if __name__ == "__main__":
    main()
