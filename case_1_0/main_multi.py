import torch
import numpy as np
import torchvision
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from Network import ConNetwork, LinearClassifier
from Loss import ContrastiveLoss
from torchvision.transforms import Compose, ToTensor, Normalize

import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def load_training_data(augment=True):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(), normalize, ])
    if augment:
        training_data = datasets.MNIST(root="../data", train=True, download=True,
                                       transform=TwoCropTransform(train_transform), )
    else:
        training_data = datasets.MNIST(root="../data", train=True, download=True,
                                       transform=Compose([ToTensor(), normalize]), )
    return training_data


def load_cifar10_test_dataloader(batch_size):
    normalize = transforms.Normalize(mean=(0.4841,), std=(0.2313,))
    train_transform = transforms.Compose(
        [transforms.Resize(size=28), transforms.Grayscale(), transforms.ToTensor(), normalize])

    test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=train_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    # data = next(iter(test_loader))
    # print(data[0][1].mean(),data[0][2].mean(),data[0][3].mean())
    # print(data[0][1].std(),data[0][2].std(),data[0][3].std())

    return test_loader


def load_mnist_test_dataloader(batch_size):
    test_data = datasets.MNIST(root="../data", train=False, download=True,
                               transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return test_loader


batch_size = 64
epochs = 10
network = ConNetwork()
loss_fn = ContrastiveLoss(temperature=0.07)
optimizer = optim.Adam(network.parameters(), lr=0.001)
train_losses = []
val_losses = []


# train encoder and projector with contrastive loss
def train_encoder(training_data):
    kfold = KFold(n_splits=5, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(training_data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=test_subsampler)
        # Run the training loop for defined number of epochs
        for epoch in range(0, epochs):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            network.train()
            iter_count, losses = 0, 0
            for batch_idx, (data, target) in enumerate(train_loader):
                images = torch.cat([data[0], data[1]], dim=0)
                outputs = network(images)
                bsz = target.shape[0]
                f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 256,2,128
                loss = loss_fn(features, target)
                losses += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_count += 1

            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, losses / iter_count))
            train_losses.append(losses / iter_count)
            torch.save(network.state_dict(), 'model_multi.pth')
            torch.save(optimizer.state_dict(), 'optimizer_multi.pth')

            with torch.no_grad():
                # Iterate over the test data and generate predictions
                iter_count, losses = 0, 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    images = torch.cat([data[0], data[1]], dim=0)
                    outputs = network(images)
                    bsz = target.shape[0]
                    f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 256,2,128

                    loss = loss_fn(features, target)
                    losses += loss.item()
                    iter_count += 1
                val_losses.append(losses / iter_count)
                print('Validation Epoch: {}\tLoss: {:.6f}'.format(epoch, losses / iter_count))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label='val')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('train_loss_multi.png')
    plt.show()


kdes = {}


def cal_kde_binary():
    num_classes = 10
    network.load_state_dict(torch.load('model_multi.pth'))
    network.eval()

    training_data = load_training_data(False)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    bandwidths = [0.0018329807108324356, 0.0018329807108324356, 0.00379269019073225, 0.0026366508987303583,
                  0.0026366508987303583, 0.0026366508987303583, 0.0026366508987303583, 0.0026366508987303583,
                  0.00379269019073225, 0.00379269019073225]
    rep_outputs = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for data, target in train_loader:
            outputs = network(data)
            for i in range(num_classes):
                idx = (target == i).nonzero(as_tuple=True)[0]
                rep_outputs[i].append(outputs[idx])

    for i in range(num_classes):
        values = torch.cat(rep_outputs[i], dim=0)
        # params = {"bandwidth": np.logspace(-4, -1, 20)}
        # grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
        # grid.fit(values)
        # kdes[i] = grid.best_estimator_
        # print('bandwidth for classes {} is {}'.format(i, grid.best_estimator_.bandwidth))

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidths[i]).fit(values)
        kdes[i] = kde


def test_multi():
    num_classes = 10
    network.load_state_dict(torch.load('model_multi.pth'))
    network.eval()

    threshold = np.log(0.005)
    test_loader = load_cifar10_test_dataloader(batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_pdfs = []
            for i in range(num_classes):
                feature_pdfs.append(kdes[i].score_samples(features))
            feature_log_pdfs = np.array(feature_pdfs)
            ## all CIFAR10 should be ood
            feature_log_pdfs[feature_log_pdfs < threshold] = np.NINF
            for i in range(feature_log_pdfs.shape[1]):
                if np.count_nonzero(feature_log_pdfs[:, i] == np.NINF) == num_classes:
                    correct += 1

    print('\nTest cifar10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total))
    test_loader = load_mnist_test_dataloader(batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_pdfs = []
            for i in range(num_classes):
                feature_pdfs.append(kdes[i].score_samples(features))
            feature_log_pdfs = np.array(feature_pdfs)
            # feature_log_pdfs[feature_log_pdfs < threshold] = np.NINF

            for i in range(feature_log_pdfs.shape[1]):
                if np.count_nonzero(feature_log_pdfs[:, i] == 1) == num_classes:
                    continue
                pre_label = np.argmax(feature_log_pdfs[:, i])
                if pre_label == target[i]:
                    correct += 1
    print('\nTest mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total))


def main():
    print('Start Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # training_data = load_training_data()
    # train_encoder(training_data)
    cal_kde_binary()
    test_multi()
    print('End Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()
