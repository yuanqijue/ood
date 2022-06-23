import torch
import math
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from Network import ConNetwork, LinearClassifier
from Loss import ContrastiveLoss

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch.nn.functional as F
import torch.optim as optim

# from torchsummary import summary

training_data = datasets.MNIST(root="data", train=True, download=True,
                               transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )

# Download test data from open datasets.
test_data = datasets.MNIST(root="data", train=False, download=True,
                           transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
print('training data length %s' % len(training_data))
print('testing data length %s' % len(test_data))

batch_size = 256  # actual batch_size for each sub projector is almost 64

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

n_epochs = 10
num_classes = 10
network = ConNetwork(num_classes)
print(network)

loss_fn = ContrastiveLoss(num_classes, temperature=0.07)
optimizer = optim.Adam(network.parameters(), lr=0.0001)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# encoder
def train_encoder(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        outputs = network(data)
        loss = loss_fn(outputs, target)
        losses = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           losses))
            train_losses.append(losses)
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


classifier = LinearClassifier(num_classes)
loss_fn_classifiers = torch.nn.BCELoss()
optimizer_classifiers = optim.Adam(network.parameters(), lr=0.1)


def train_cls(epoch):
    network.load_state_dict(torch.load('results/model.pth'))
    network.eval()
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.no_grad():
            features = network.encoder(data)
        optimizer_classifiers.zero_grad()
        outputs = classifier(features)
        losses = 0

        for i in range(num_classes):
            labels = torch.eq(target, i).float()
            loss = loss_fn_classifiers(outputs[i], labels.view(-1, 1))
            losses += loss.item()
            if i == num_classes - 1:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
        optimizer_classifiers.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), losses,
                                                                           accuracy(outputs, target)))
            train_losses.append(losses)
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(classifier.state_dict(), 'results/model_classifiers.pth')
            torch.save(optimizer_classifiers.state_dict(), 'results/optimizer_classifiers.pth')


def test():
    network.eval()
    classifier.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        outputs = []
        with torch.no_grad():
            features = network.encoder(data)
            for i in range(num_classes):
                output = classifier(features)
                outputs.append(output)
        print('Accuracy', accuracy(outputs, target))


def accuracy(outputs, labels):
    outputs = outputs.view(num_classes, -1)
    pred_labels = torch.argmax(outputs.T, dim=1)
    num_corrects = torch.eq(pred_labels, labels).sum().float().item()
    return num_corrects / labels.shape[0]


def main():
    # for epoch in range(1, 5):
    #     train_encoder(epoch)
    for epoch in range(1, 5):
        train_cls(epoch)


if __name__ == '__main__':
    main()
