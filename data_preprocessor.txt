import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from typing import *
import h5py


class NeuromorphicPreprocessor:
    def __init__(self,
                 dataset_name: str,
                 time_steps: int = 20,
                 encoding: str = 'rate',
                 event_bins: int = 10,
                 img_size: Tuple[int, int] = (128, 128)):
        """预处理模块"""
        self.dataset_name = dataset_name.lower()
        self.time_steps = time_steps
        self.encoding = encoding
        self.event_bins = event_bins
        self.img_size = img_size

        self._init_dataset_specific_params()

    def _init_dataset_specific_params(self):
        self.dataset_config = {
            # 静态数据集配置
            'mnist': {
                'norm_mean': (0.1307,),
                'norm_std': (0.3081,),
                'channels': 1
            },
            'cifar10': {
                'norm_mean': (0.4914, 0.4822, 0.4465),
                'norm_std': (0.2023, 0.1994, 0.2010),
                'channels': 3
            },
            'cifar100': {
                'norm_mean': (0.5071, 0.4867, 0.4408),
                'norm_std': (0.2675, 0.2565, 0.2761),
                'channels': 3
            },
            # 事件数据集配置
            'cifar10-dvs': {
                'event_scale': 0.8,
                'polarity': True
            },
            'dvs-gesture': {
                'event_scale': 1.0,
                'polarity': False
            },
            'nmnist': {
                'event_scale': 0.6,
                'polarity': True
            }
        }

        # 校验
        if self.dataset_name not in self.dataset_config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _get_static_transform(self) -> transforms.Compose:
        """静态数据集预处理"""
        cfg = self.dataset_config[self.dataset_name]
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(cfg['norm_mean'], cfg['norm_std'])
        ])

    def _get_event_transform(self) -> transforms.Compose:
        """事件数据集预处理"""
        return transforms.Compose([
            EventToFrame(bins=self.event_bins),
            transforms.Resize(self.img_size),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: x / 255.0)
        ])

    def _load_event_dataset(self, root: str) -> Dataset:
        """加载事件数据集"""
        if self.dataset_name == 'cifar10-dvs':
            return CIFAR10DVS(root=root, transform=self._get_event_transform())
        elif self.dataset_name == 'dvs-gesture':
            return DVSGesture(root=root, transform=self._get_event_transform())
        elif self.dataset_name == 'nmnist':
            return NMNIST(root=root, transform=self._get_event_transform())
        else:
            raise NotImplementedError(f"Loader for {self.dataset_name} not implemented")

    def get_loader(self,
                   batch_size: int = 32,
                   split: str = 'train') -> DataLoader:
        """获取数据加载器"""
        # 静态数据集处理
        if self.dataset_name in ['mnist', 'cifar10', 'cifar100']:
            transform = self._get_static_transform()
            # 编码处理
            dataset = self._load_static_dataset(transform)
            dataset.data = self._static_encoding(dataset.data)
        # 事件数据集处理
        else:
            dataset = self._load_event_dataset(os.path.join('./data', self.dataset_name))

        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

    def _load_static_dataset(self, transform: callable) -> Dataset:
        """加载静态数据集"""
        if self.dataset_name == 'mnist':
            return datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        elif self.dataset_name == 'cifar10':
            return datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        elif self.dataset_name == 'cifar100':
            return datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
        else:
            raise ValueError("Invalid static dataset name")