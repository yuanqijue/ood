import os
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from model import VAE
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Generic runner for VAE models')

parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='path to the config file',
                    default='conf.yaml')

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'], name=config['model_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = VAE(**config['model_params'])
path = 'logs/VAE_CL/version_42/checkpoints/last.ckpt'
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path=path, vae_model=model, params=config['exp_params'])
network = experiment.model
network.eval()

batch_size = config['data_params']['val_batch_size']
train_data = datasets.MNIST(root="../data", train=True, download=True,
                            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
train_loader = DataLoader(train_data, batch_size=batch_size)

test_data = datasets.MNIST(root="../data", train=False, download=True,
                           transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), )
test_loader = DataLoader(test_data, batch_size=batch_size)

threshold = 0
num_samples = 128

# train_max_log_px = -793.9294311346123

correct, total = 0, 0
with torch.no_grad():
    for data, target in train_loader:
        total += target.shape[0]
        log_px = network.log_marginal_likelihood(data, num_samples)
        print('p(x) in train MNIST', np.mean(log_px))
        if total >= batch_size * 10:
            break

# print('Maximum log p(x) is:', train_max_log_px)

correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        log_px = network.log_marginal_likelihood(data, num_samples)
        print('p(x) in test MNIST', np.mean(log_px))
        if total >= batch_size * 10:
            break

# print('\nTest mnist set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, total, 100. * correct / total))

train_transform = transforms.Compose([transforms.Resize(size=28), transforms.Grayscale(), transforms.ToTensor(),
                                      Normalize(mean=(0.4841,), std=(0.2313,))])
test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=train_transform)
test_loader = DataLoader(test_data, batch_size=batch_size)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        total += target.shape[0]
        log_px = network.log_marginal_likelihood(data, num_samples)
        print('p(x) in test CIFAR10', np.mean(log_px))
        if total >= batch_size * 10:
            break
