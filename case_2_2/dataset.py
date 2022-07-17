import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.utils import data
import zipfile


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(self, data_path: str, train_batch_size: int = 8, val_batch_size: int = 8,
                 patch_size: Union[int, Sequence[int]] = (256, 256), num_workers: int = 0, pin_memory: bool = False,
                 **kwargs, ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # cifar10
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(),
             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
             transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), normalize, ])

        self.train_dataset = datasets.CIFAR10(root="data", train=True, download=True,
                                              transform=TwoCropTransform(train_transform), )

        # use 20% of training data for validation
        train_set_size = int(len(self.train_dataset) * 0.8)
        valid_set_size = len(self.train_dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = data.random_split(self.train_dataset, [train_set_size, valid_set_size],
                                                                 generator=seed)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=self.pin_memory, )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory, )
