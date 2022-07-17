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
path = 'logs/VAE_CL/version_42/checkpoints/last.ckpt'
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

threshold = 0
num_samples = 128

correct, total = 0, 0
num_classes = 10
rep_outputs = {i: [] for i in range(num_classes)}
kdes = {}
with torch.no_grad():
    for data, target in train_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        for i in range(num_classes):
            idx = (target == i).nonzero(as_tuple=True)[0]
            rep_outputs[i].append(mu[idx])

for i in range(num_classes):
    values = torch.cat(rep_outputs[i], dim=0)
    # params = {"bandwidth": np.logspace(-4, -1, 20)}
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
    # grid.fit(values)
    # kdes[i] = grid.best_estimator_
    # print('bandwidth for classes {} is {}'.format(i, grid.best_estimator_.bandwidth))

    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(values)
    kdes[i] = kde

# mu_data = torch.cat(mu_data, dim=0).numpy()
# kmeans = KMeans(n_clusters=10, random_state=0).fit(mu_data)
# print(kmeans.cluster_centers_)

threshold = np.log(0.001)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        # print(kmeans.fit_transform(mu))
        feature_pdfs = []
        for i in range(num_classes):
            feature_pdfs.append(kdes[i].score_samples(mu.numpy()))
        feature_log_pdfs = np.array(feature_pdfs)
        feature_log_pdfs[feature_log_pdfs < threshold] = np.NINF

        for i in range(feature_log_pdfs.shape[1]):
            if np.count_nonzero(feature_log_pdfs[:, i] == np.NINF) == num_classes:
                continue
            pre_label = np.argmax(feature_log_pdfs[:, i])
            if pre_label == target[i]:
                correct += 1
with open('out.txt', 'a') as f:
    print('\nTest MNIST set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total), file=f)

train_transform = transforms.Compose([transforms.Resize(size=28), transforms.Grayscale(), transforms.ToTensor(),
                                      Normalize(mean=(0.4841,), std=(0.2313,))])
test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=train_transform)
test_loader = DataLoader(test_data, batch_size=batch_size)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        feature_pdfs = []
        for i in range(num_classes):
            feature_pdfs.append(kdes[i].score_samples(mu.numpy()))
        feature_log_pdfs = np.array(feature_pdfs)
        ## all CIFAR10 should be ood
        feature_log_pdfs[feature_log_pdfs < threshold] = np.NINF
        for i in range(feature_log_pdfs.shape[1]):
            if np.count_nonzero(feature_log_pdfs[:, i] == np.NINF) == num_classes:
                correct += 1
with open('out.txt', 'a') as f:
    print('\nTest CIFAR10 set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total), file=f)
