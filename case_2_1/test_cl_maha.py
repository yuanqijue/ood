import torch
import numpy as np
from model import VAE
from experiment import VAEXperiment
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

model = VAE(in_channels=1, latent_dim=128, hidden_dims=[32, 64, 128, 265])
path = 'logs/VAE_CL/version_39/checkpoints/last.ckpt'
exp_params = {'LR': 0.001, 'weight_decay': 0.0, 'kld_weight': 0.00025, 'manual_seed': 1000}
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path=path, vae_model=model, params=exp_params)
network = experiment.model
network.eval()

batch_size = 64
train_data = datasets.MNIST(root="../data", train=True, download=True,
                            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
train_loader = DataLoader(train_data, batch_size=batch_size)

test_data = datasets.MNIST(root="../data", train=False, download=True,
                           transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
test_loader = DataLoader(test_data, batch_size=batch_size)


def calc_mahalanobis_distance(input_sample, mean, inv_cov):
    input_sample = input_sample.numpy() - mean
    mahal = input_sample @ inv_cov @ input_sample.T
    return mahal.diagonal()


threshold = 0
num_samples = 128

correct, total = 0, 0
num_classes = 10
rep_outputs = {i: [] for i in range(num_classes)}
with torch.no_grad():
    for data, target in train_loader:
        [x_hat, x, mu, log_var, z] = network(data)
        for i in range(num_classes):
            idx = (target == i).nonzero(as_tuple=True)[0]
            rep_outputs[i].append(mu[idx])

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

#################### test
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        features = mu
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
with open('out.txt', 'a') as f:
    print('\nMahalanobis distance Test mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                      100. * correct / total), file=f)

train_transform = transforms.Compose([transforms.Resize(size=28), transforms.Grayscale(), transforms.ToTensor(),
                                      Normalize(mean=(0.4841,), std=(0.2313,))])
test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=train_transform)
test_loader = DataLoader(test_data, batch_size=batch_size)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        features = mu
        feature_distances = []
        for i in range(num_classes):
            dis = calc_mahalanobis_distance(features, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy()
            dis[dis > threshold[i]] = np.Inf
            feature_distances.append(dis)
        feature_distances = np.array(feature_distances)
        for i in range(feature_distances.shape[1]):
            if np.count_nonzero(feature_distances[:, i] == np.Inf) == num_classes:
                correct += 1
with open('out.txt', 'a') as f:
    print('\nMahalanobis distance Test cifar10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total,
                                                                                        100. * correct / total), file=f)
