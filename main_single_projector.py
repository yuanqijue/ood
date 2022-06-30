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


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


batch_size = 256  # actual batch_size for each sub projector is almost 64

normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(),
     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2),
     transforms.ToTensor(), normalize, ])

training_data = datasets.MNIST(root="data", train=True, download=True, transform=TwoCropTransform(train_transform), )

# Download test data from open datasets.
test_data = datasets.MNIST(root="data", train=False, download=True, transform=TwoCropTransform(train_transform), )
# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


##### data prepared for classifier
classifier_training_data = datasets.MNIST(root="data", train=True, download=True,
                               transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
# Download test data from open datasets.
classifier_test_data = datasets.MNIST(root="data", train=False, download=True,
                           transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
classifier_test_loader = DataLoader(classifier_test_data, batch_size=batch_size)
# print('training data length %s' % len(training_data))
# print('testing data length %s' % len(test_data))


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


classifier = LinearClassifier(num_classes)
loss_fn_classifiers = torch.nn.BCELoss()

# Initialize optimizer
optimizer_classifiers = optim.Adam(classifier.parameters(), lr=0.001)

train_cls_acc = []
val_cls_acc = []
train_cls_losses = []
val_cls_losses = []

from sklearn.model_selection import KFold

k_folds = 5


def train_cls():
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    n_cls_epochs = 5

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(classifier_training_data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(classifier_training_data, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(classifier_training_data, batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        network.load_state_dict(torch.load('results/model_single.pth'))
        network.eval()
        classifier.train()

        # Run the training loop for defined number of epochs
        for epoch in range(0, n_cls_epochs):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            # Set current loss value
            current_loss = 0.0
            correct_count = 0
            # Iterate over the DataLoader for training data
            iterate_count, total = 0, 0
            for j, (data, target) in enumerate(trainloader):
                iterate_count += 1
                with torch.no_grad():
                    features = network.encoder(data)
                optimizer_classifiers.zero_grad()
                outputs = classifier(features)
                losses = 0
                total += target.size(0)
                for i in range(num_classes):
                    labels = torch.eq(target, i).float()
                    loss = loss_fn_classifiers(outputs[i], labels.view(-1, 1))
                    losses += loss.item()
                    if i == num_classes - 1:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)
                optimizer_classifiers.step()
                current_loss += losses
                correct_count += accuracy_count(outputs, target)

            train_cls_acc.append(correct_count / total)
            train_cls_losses.append(current_loss / iterate_count)
            # Saving the model
            torch.save(classifier.state_dict(), 'results/model_classifiers_single.pth')
            torch.save(optimizer_classifiers.state_dict(), 'results/optimizer_classifiers_single.pth')

            # Evaluation for this fold
            correct, total = 0, 0
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, (data, target) in enumerate(testloader):
                    features = network.encoder(data)
                    outputs = classifier(features)
                    # Set total and correct
                    total += target.size(0)
                    correct += accuracy_count(outputs, target)

            val_cls_acc.append(correct / total)
            # Print accuracy
            print('Accuracy for fold {}: {:.4f} \n'.format(fold, 100.0 * correct / total))
    plt.plot(train_cls_acc, label="train")
    plt.plot(val_cls_acc, label='val')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('classifier_train_accuracy_single.png')
    plt.show()

    plt.plot(train_cls_losses, label="train")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('classifier_train_loss_single.png')
    plt.show()


import os
import glob


def test():
    files = glob.glob('wrong_imgs/*')
    for f in files:
        os.remove(f)
    network.load_state_dict(torch.load('results/model_single.pth'))
    classifier.load_state_dict(torch.load('results/model_classifiers_single.pth'))
    network.eval()
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for data, target in classifier_test_loader:
            features = network.encoder(data)
            outputs = classifier(features)
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
    test()  # 98.5500%
    print('End Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()
