import torch
import numpy as np
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
# print('training data length %s' % len(training_data))
# print('testing data length %s' % len(test_data))

batch_size = 256  # actual batch_size for each sub projector is almost 64

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

num_classes = 10
network = ConNetwork(num_classes)
# print(network)

loss_fn = ContrastiveLoss(num_classes, temperature=0.07)
optimizer = optim.Adam(network.parameters(), lr=0.0001)

train_losses = []


# encoder
def train_encoder():
    n_encoder_epochs = 5
    for epoch in range(1, n_encoder_epochs + 1):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            outputs = network(data)
            loss = loss_fn(outputs, target)
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
                torch.save(network.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')
    plt.plot(train_losses, label="train")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('encoder_train_loss.png')
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
    n_cls_epochs = 10

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(training_data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        network.load_state_dict(torch.load('results/model.pth'))
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
            torch.save(classifier.state_dict(), 'results/model_classifiers.pth')
            torch.save(optimizer_classifiers.state_dict(), 'results/optimizer_classifiers.pth')

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
    plt.savefig('classifier_train_accuracy.png')
    plt.show()

    plt.plot(train_cls_losses, label="train")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('classifier_train_loss.png')
    plt.show()


def test():
    network.load_state_dict(torch.load('results/model.pth'))
    classifier.load_state_dict(torch.load('results/model_classifiers.pth'))
    network.eval()
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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

    if data:
        error_index_samples = ((count_one_tensor != 1).nonzero(as_tuple=True)[0])

    return num_corrects


def accuracy(outputs, labels):
    return accuracy_count(outputs, labels) / labels.shape[0]


def main():
    # train_encoder()
    # train_cls()

    test()  # 97.9300% 165个存在多个分类，42个分类错误
    print('error count', np.array(ood_error).sum())


if __name__ == '__main__':
    main()
