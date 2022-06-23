import torch
from torch import nn
import torch.nn.functional as F


class ConNetwork(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes):
        super(ConNetwork, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2))

        self.projs = nn.ModuleList()

        for i in range(num_classes):
            cls = nn.Sequential(nn.Linear(32 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 64))
            self.projs.append(cls)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 5 * 5)
        return x

    def forward(self, x):
        x = self.encoder(x)
        out = []
        for i in range(self.num_classes):
            out.append(F.normalize(self.projs[i](x), dim=1))
        return out


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList()
        for i in range(num_classes):
            cls = nn.Sequential(nn.Linear(32 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
            self.classifiers.append(cls)

    def forward(self, x):
        out = []
        for i in range(self.num_classes):
            out.append(self.classifiers[i](x).unsqueeze(0))
        return torch.cat(out, dim=0)
