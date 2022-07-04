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
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


## 将10个分类的数据转化成2个分类，其中positive_number表示正样本数字，其他为负样本数字
def get_balanced_one_hot_data_indices(num_classes, positive_number, labels):
    positive_indices = np.where(labels == positive_number)[0]
    negative_indices = np.array([], dtype=np.dtype(int))
    for i in range(num_classes):
        if i == positive_number:
            continue
        size = len(np.where(labels == i)[0])
        scaled_size = size // (num_classes - 1)
        sub_idx = np.where(labels == i)[0]
        np.random.shuffle(sub_idx)
        sub_idx = sub_idx[:scaled_size]
        negative_indices = np.concatenate([negative_indices, sub_idx], axis=0, dtype=np.dtype(int))
    return positive_indices, negative_indices


def load_binary_training_data(positive_number, augment=True):
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
    positive_indices, negative_indices = get_balanced_one_hot_data_indices(10, positive_number,
                                                                           training_data.targets.numpy())
    training_data.targets[positive_indices] = 1
    training_data.targets[negative_indices] = 0

    train_indices = np.concatenate((positive_indices, negative_indices), axis=0)
    train_data = torch.utils.data.dataset.Subset(training_data, train_indices)
    return train_data


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


def load_mnist_test_dataloader(batch_size, positive_number):
    test_data = datasets.MNIST(root="../data", train=False, download=True,
                               transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
    positive_indices, negative_indices = get_balanced_one_hot_data_indices(10, positive_number,
                                                                           test_data.targets.numpy())
    test_data.targets[positive_indices] = 1
    test_data.targets[negative_indices] = 0
    train_indices = np.concatenate((positive_indices, negative_indices), axis=0)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    return test_loader


positive_number = 0
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
            torch.save(network.state_dict(), 'model_binary.pth')
            torch.save(optimizer.state_dict(), 'optimizer_binary.pth')

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
    plt.savefig('train_loss.png')
    plt.show()


kdes = {}


def cal_kde_binary():
    num_classes = 2
    network.load_state_dict(torch.load('model_binary.pth'))
    network.eval()

    training_data = load_binary_training_data(positive_number, False)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

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
        kde = KernelDensity(kernel='gaussian', bandwidth=0.00061585).fit(values)
        kdes[i] = kde


def test_binary():
    network.load_state_dict(torch.load('model_binary.pth'))
    network.eval()

    threshold = np.log(0.005)
    test_loader = load_cifar10_test_dataloader(batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_pdfs = []
            for i in range(2):
                feature_pdfs.append(kdes[i].score_samples(features))
            feature_log_pdfs = np.array(feature_pdfs)
            ## all CIFAR10 should be ood
            feature_log_pdfs[feature_log_pdfs < threshold] = 1
            for i in range(feature_log_pdfs.shape[1]):
                cls_0 = feature_log_pdfs[0][i]
                cls_1 = feature_log_pdfs[1][i]
                if cls_0 == 1 and cls_1 == 1:
                    correct += 1

    print('\nTest cifar10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total))
    test_loader = load_mnist_test_dataloader(batch_size, positive_number)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_pdfs = []
            for i in range(2):
                feature_pdfs.append(kdes[i].score_samples(features))
            feature_log_pdfs = np.array(feature_pdfs)
            feature_log_pdfs[feature_log_pdfs < threshold] = 1

            # pre_labels = np.argmax(feature_log_pdfs, axis=0)
            # correct += np.count_nonzero(pre_labels == target.numpy())

            for i in range(feature_log_pdfs.shape[1]):
                cls_0 = feature_log_pdfs[0][i]
                cls_1 = feature_log_pdfs[1][i]
                if cls_0 == 1 and cls_1 == 1:
                    continue
                if cls_0 != 1 and cls_1 != 1 and np.argmax(feature_log_pdfs[:, i]) == target[i]:
                    correct += 1
                if cls_0 == 1 and target[i] == 1:
                    correct += 1
                if cls_1 == 1 and target[i] == 0:
                    correct += 1
    print('\nTest mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total))


def cal_mahalanobis_mean_icov():
    num_classes = 2
    network.load_state_dict(torch.load('model_binary.pth'))
    network.eval()

    training_data = load_binary_training_data(positive_number, False)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    rep_outputs = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for data, target in train_loader:
            outputs = network(data)
            for i in range(num_classes):
                idx = (target == i).nonzero(as_tuple=True)[0]
                rep_outputs[i].append(outputs[idx])

    mean_icovs = {i: [] for i in range(num_classes)}
    for i in range(num_classes):
        values = torch.cat(rep_outputs[i], dim=0)
        data = values.numpy()
        mean = np.mean(data, axis=0)
        inv_cov = np.linalg.inv(np.cov(data.T))
        mean_icovs[i] = {'mean': mean, 'inv_cov': inv_cov}

    threshold = []
    factor = 0
    for i in range(num_classes):
        values = torch.cat(rep_outputs[i], dim=0)
        distance = calc_mahalanobis_distance(values, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov'])
        distance = np.sort(distance)
        threshold_dis = distance[-1] + factor * distance[-1]
        threshold.append(threshold_dis)
    print('Mahalanobis distance threshold for class 0 is {}, for class 1 is {}'.format(threshold[0], threshold[1]))
    return mean_icovs, threshold


def test_binary_mahalanobis(mean_icovs, threshold):
    num_classes = 2
    network.load_state_dict(torch.load('model_binary.pth'))
    network.eval()

    test_loader = load_cifar10_test_dataloader(batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_distances = []
            for i in range(num_classes):
                dis = calc_mahalanobis_distance(features, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy()
                dis[dis > threshold[i]] = np.Inf
                feature_distances.append(dis)
            feature_distances = np.array(feature_distances)
            for i in range(feature_distances.shape[1]):
                if np.count_nonzero(feature_distances[:, i] == np.Inf) == num_classes:
                    correct += 1

    print('\nMahalanobis distance Test cifar10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                        100. * correct / total))
    test_loader = load_mnist_test_dataloader(batch_size, positive_number)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            feature_distances = []
            for i in range(num_classes):
                dis = calc_mahalanobis_distance(features, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy()
                dis[dis > threshold[i]] = np.Inf
                dis = dis - threshold[i]
                feature_distances.append(dis)
            feature_distances = np.array(feature_distances)
            for i in range(feature_distances.shape[1]):
                if np.count_nonzero(feature_distances[:, i] == np.Inf) == num_classes:
                    continue
                pre_label = np.argmin(feature_distances[:, i])
                if pre_label == target[i]:
                    correct += 1
    print('\nMahalanobis distance Test mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                      100. * correct / total))


def test_binary_mahalanobis_only_positive(mean_icovs, threshold):
    num_classes = 2
    network.load_state_dict(torch.load('model_binary.pth'))
    network.eval()

    test_loader = load_cifar10_test_dataloader(batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            dis = calc_mahalanobis_distance(features, mean_icovs[1]['mean'], mean_icovs[1]['inv_cov']).copy()
            dis[dis > threshold[1]] = np.Inf
            correct += np.count_nonzero(dis == np.Inf)

    print('\nMahalanobis distance Test cifar10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                        100. * correct / total))
    test_loader = load_mnist_test_dataloader(batch_size, positive_number)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            total += target.shape[0]
            features = network(data)
            dis = calc_mahalanobis_distance(features, mean_icovs[1]['mean'], mean_icovs[1]['inv_cov']).copy()
            # p_value = 1 - chi2.cdf(dis, 127)
            # p_value[p_value < 0.001] = np.Inf

            dis[dis > threshold[1]] = np.Inf
            for i in range(dis.shape[0]):
                if dis[i] == np.Inf and target[i] == 0:
                    correct += 1
                if dis[i] != np.Inf and target[i] == 1:
                    correct += 1
    print('\nMahalanobis distance Test mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                      100. * correct / total))


def calc_mahalanobis_distance(input_sample, mean, inv_cov):
    input_sample = input_sample.numpy() - mean
    mahal = input_sample @ inv_cov @ input_sample.T
    return mahal.diagonal()


def main():
    print('Start Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # training_data = load_binary_training_data(positive_number)
    # train_encoder(training_data)
    # cal_kde_binary()
    # test_binary()
    # test(2)
    mean_icovs, threshold = cal_mahalanobis_mean_icov()
    # test_binary_mahalanobis(mean_icovs, threshold)
    test_binary_mahalanobis_only_positive(mean_icovs, threshold)

    print('End Time', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()
