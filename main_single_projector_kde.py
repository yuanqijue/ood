import torch
import numpy as np
import torchvision
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from Network_single_projector import ConNetwork, LinearClassifier
from Loss_single_projector import ContrastiveLoss

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import KernelDensity
from scipy import stats


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(),
     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2),
     transforms.ToTensor(), normalize, ])

training_data = datasets.MNIST(root="data", train=True, download=True, transform=TwoCropTransform(train_transform), )

# Download test data from open datasets.
test_data = datasets.MNIST(root="data", train=False, download=True, transform=TwoCropTransform(train_transform), )

classifier_training_data = datasets.MNIST(root="data", train=True, download=True,
                                          transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )

# Download test data from open datasets.
classifier_test_data = datasets.MNIST(root="data", train=False, download=True,
                                      transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )

# print('training data length %s' % len(training_data))
# print('testing data length %s' % len(test_data))

batch_size = 256  # actual batch_size for each sub projector is almost 64

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

num_classes = 10
network = ConNetwork(num_classes)
# print(network)

loss_fn = ContrastiveLoss(temperature=0.07)
optimizer = optim.Adam(network.parameters(), lr=0.0001)

train_losses = []


# encoder
def train_encoder():
    n_encoder_epochs = 10
    for epoch in range(1, n_encoder_epochs + 1):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            images = torch.cat([data[0], data[1]], dim=0)
            outputs = network(images)
            bsz = target.shape[0]
            f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 256,2,128

            loss = loss_fn(features, target)
            losses = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(losses)

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               losses))
                torch.save(network.state_dict(), 'results/model_single.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer_single.pth')
    plt.plot(train_losses, label="train")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('encoder_train_loss_single.png')
    plt.show()


kdes = {}

def cal_kde():
    network.load_state_dict(torch.load('results/model_single.pth'))
    network.eval()

    rep_outputs = { i:[] for i in range(num_classes)}
    with torch.no_grad():
        for data, target in test_loader:
            outputs = network(data)
            for i in range(num_classes):
                idx = (target == i).nonzero().detach()
                rep_outputs[i].append(outputs[idx])

    for i in range(num_classes):
        values = torch.cat(rep_outputs[i],dim=0)
        n = values.shape[0]
        # Modified Silverman's rule of thumb
        IQR = np.subtract(*np.percentile(values, [75, 25], axis=0))
        sigma_est = np.std(values, axis=0)
        h = np.clip(0.9 * min(sigma_est, IQR / 1.34) * n ** (-1 / 5), 1e-5, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(values)
        # kdes[i] = stats.gaussian_kde(values)

import os
import glob


def test():
    files = glob.glob('wrong_imgs/*')
    for f in files:
        os.remove(f)
    network.load_state_dict(torch.load('results/model_single.pth'))
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            features = network.encoder(data)
            correct += accuracy_count(outputs, target, data)
    print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))


ood_error = []


def accuracy_count(outputs, labels, data=None):
    outputs = outputs.view(num_classes, -1).T
    outputs = torch.round(outputs)
    one_tensor = torch.eq(outputs, 1).long()
    count_one_tensor = torch.count_nonzero(one_tensor, dim=1)
    index_samples = ((count_one_tensor == 1).nonzero(as_tuple=True)[0])
    ood_error.append(outputs.shape[0] - len(index_samples))
    pred_labels = torch.argmax(outputs[index_samples], dim=1)
    num_corrects = torch.eq(pred_labels, labels[index_samples]).sum().float().item()

    if debug and type(data) != type(None):
        error_index_samples = ((count_one_tensor != 1).nonzero(as_tuple=True)[0])
        error_imgs = data[error_index_samples]
        print('Detecting {} error images'.format(len(error_index_samples)))

        for show_index, img in enumerate(error_imgs):
            show_label_index = error_index_samples[show_index]
            pred_labels = ((outputs[show_label_index] == 1).nonzero(as_tuple=True)[0])
            im = Image.fromarray((error_imgs[show_index][0].numpy() * 0.3081 + 0.1307) * 255).convert('L')
            im.save("wrong_imgs/Actual label is {}, Pred label is {}.png".format(labels[show_label_index],
                                                                                 pred_labels.numpy()))  # print('Actual label is {}, Pred label is {}'.format(labels[show_label_index], pred_labels))  # plt.imshow(error_imgs[show_index][0], cmap='gray', interpolation='none')  # plt.show()
    return num_corrects


def accuracy(outputs, labels):
    return accuracy_count(outputs, labels) / labels.shape[0]


import time

debug = True


def main():
    print('Start Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # train_encoder()
    train_cls()

    # test()  # 97.9300% 165个存在多个分类，42个分类错误
    # 优化后 98.6900% 131个异常图片中，存在多个分类或者有32个
    print('End Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()
