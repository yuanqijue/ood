import torch
import numpy as np
from model import VAE
from experiment import VAEXperiment
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

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


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = roc_auc_score(labels, data)
    return auroc


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

# threshold = []
# factor = 0
# for i in range(num_classes):
#     values = torch.cat(rep_outputs[i], dim=0)
#     distance = calc_mahalanobis_distance(values, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov'])
#     distance = np.sort(distance)
#     threshold_dis = distance[-1] + factor * distance[-1]
#     threshold.append(threshold_dis)
# print('Mahalanobis distance threshold for class 0 is {}, for class 1 is {}'.format(threshold[0], threshold[1]))

#################### test
correct, total = 0, 0
in_scores = []
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        feature_distances = np.array(
            [calc_mahalanobis_distance(mu, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy() for i in
             range(num_classes)]).reshape(-1, num_classes)
        min_idx = np.argmin(feature_distances, axis=1)
        in_scores.append([dist[min_idx[i]] for i, dist in enumerate(feature_distances)])
in_scores = np.concatenate(in_scores)

train_transform = transforms.Compose([transforms.Resize(size=28), transforms.Grayscale(), transforms.ToTensor(),
                                      Normalize(mean=(0.4841,), std=(0.2313,))])
test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=train_transform)
test_loader = DataLoader(test_data, batch_size=batch_size)
correct, total = 0, 0
ood_scores = []
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        feature_distances = np.array(
            [calc_mahalanobis_distance(mu, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy() for i in
             range(num_classes)]).reshape(-1, num_classes)
        min_idx = np.argmin(feature_distances, axis=1)
        ood_scores.append([dist[min_idx[i]] for i, dist in enumerate(feature_distances)])
ood_scores = np.concatenate(ood_scores)

auroc_ssd = get_roc_sklearn(in_scores, ood_scores)
print("AUROC {0}".format(auroc_ssd))
