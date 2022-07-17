import torch
import yaml
import argparse
import numpy as np
from model import VAE
from experiment import VAEXperiment
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generic test for VAE models')

parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='path to the config file',
                    default='conf.yaml')

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = VAE(**config['model_params'])

path = 'logs/VAE_CL_CIFAR10/version_1/checkpoints/last.ckpt'
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path=path, vae_model=model, params=config['exp_params'])
network = experiment.model
network.eval()

batch_size = 64
normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=Compose([ToTensor(), normalize]), )
train_loader = DataLoader(train_data, batch_size=batch_size)

test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=Compose([ToTensor(), normalize]), )
test_loader = DataLoader(test_data, batch_size=batch_size)

test_ood_data = datasets.CIFAR100(root="data", train=False, download=True, transform=Compose(
    [ToTensor(), Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))]))
test_ood_loader = DataLoader(test_ood_data, batch_size=batch_size)


def calc_mahalanobis_distance(input_sample, mean, inv_cov):
    input_sample = input_sample.numpy() - mean
    mahal = input_sample @ inv_cov @ input_sample.T
    return mahal.diagonal()


###########
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

#################### test id
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

#################### test ood
correct, total = 0, 0
ood_scores = []
with torch.no_grad():
    for data, target in test_ood_loader:
        total += target.shape[0]
        [x_hat, x, mu, log_var, z] = network(data)
        feature_distances = np.array(
            [calc_mahalanobis_distance(mu, mean_icovs[i]['mean'], mean_icovs[i]['inv_cov']).copy() for i in
             range(num_classes)]).reshape(-1, num_classes)
        min_idx = np.argmin(feature_distances, axis=1)
        ood_scores.append([dist[min_idx[i]] for i, dist in enumerate(feature_distances)])
ood_scores = np.concatenate(ood_scores)

labels = [0] * len(in_scores) + [1] * len(ood_scores)
data = np.concatenate((in_scores, ood_scores))
auroc = roc_auc_score(labels, data)

print("AUROC {0}".format(auroc))

# calculate roc curves
fpr, tpr, _ = roc_curve(labels, data)
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.', label='VAE with CL, kld_weight: 1')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
