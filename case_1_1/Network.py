import torch
from torch import nn
import torch.nn.functional as F


class ConNetwork(nn.Module):
    """Linear classifier"""

    def __init__(self):
        super(ConNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2))

        self.projector = nn.Sequential(nn.Linear(128 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 128))

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 5 * 5)
        return x

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(self.projector(x), dim=1)


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList()
        for i in range(num_classes):
            cls = nn.Sequential(nn.Linear(128 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
            self.classifiers.append(cls)

    def forward(self, x):
        out = []
        for i in range(self.num_classes):
            out.append(self.classifiers[i](x).unsqueeze(0))
        return torch.cat(out, dim=0)
