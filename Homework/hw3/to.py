from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

# defining FC layers
Model = torch.nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 50),
    nn.ReLU(),
    nn.Linear(50, 10))


# defining training progress
def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        data = data.view(-1, 784)
        output = model(data)
        if loss == 'CE':
            # CrossEntropy Loss
            loss_fn = nn.CrossEntropyLoss()
        if loss == 'MSE':
            # MSE Loss
            y_onehot = target.numpy()
            y_onehot = (np.arange(10) == y_onehot[:, None]).astype(np.float32)
            target = torch.from_numpy(y_onehot)
            loss_fn = nn.MSELoss()
        loss_ = loss_fn(output, target)
        loss_.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_.item()))


# defining testing
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # sum up batch loss,negative log likelihood loss

            pred = output.argmax(dim=1, keepdim=True)
            # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# main function
def main():
    device = torch.device("cuda")
    # load MNIST dataset
    batch_size = 128
    test_batch_size = 10000
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)
    # set optimizer
    lr = 0.01
    model = Model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    time0 = time.time()
    # Training settings
    epochs = 10
    loss = 'CE'
    # start training
    time0 = time.time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, loss, epoch)
        test(model, device, test_loader)
    time1 = time.time()
    print('Training and Testing total excution time is: %s seconds ' % (time1 - time0))


if __name__ == '__main__':
    main()
