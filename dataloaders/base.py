import os

import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset


class FastCelebA(Dataset):
    def __init__(self, data, attr):
        self.dataset = data
        self.attr = attr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.attr[index]


def CelebA(root, skip_normalization=False, train_aug=False, image_size=64, target_type='attr'):
    transform = transforms.Compose([

        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CelebA(root=root, download=True, transform=transform,
                                          target_type=target_type)
    print("Loading data")
    save_path = f"{root}/fast_celeba"
    if os.path.exists(save_path):
        fast_celeba = torch.load(save_path)
    else:
        train_loader = DataLoader(dataset, batch_size=len(dataset))
        data = next(iter(train_loader))
        fast_celeba = FastCelebA(data[0], data[1])
        torch.save(fast_celeba, save_path)
    # train_set = CacheClassLabel(train_set)
    # val_set = CacheClassLabel(val_set)
    return fast_celeba, None, 64, 3


def MNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 28, 1


def Omniglot(dataroot, skip_normalization=False, train_aug=False):
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(1, -1)
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(1, -1)
        ])

    train_transform = val_transform

    train_dataset = torchvision.datasets.Omniglot(
        root=dataroot,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    # val_dataset = torchvision.datasets.MNIST(
    #     dataroot,
    #     train=False,
    #     transform=val_transform
    # )
    # val_dataset = CacheClassLabel(val_dataset)
    print("Using train dataset for validation in OMNIGLOT")
    return train_dataset, train_dataset, 28, 1


def FashionMNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))  # for  28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 28, 1


def DoubleMNIST(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for  28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset_fashion = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    # train_dataset_fashion = CacheClassLabel(train_dataset_fashion)

    train_dataset_mnist = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    # train_dataset_mnist = CacheClassLabel(train_dataset_mnist)

    val_dataset_fashion = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    # val_dataset_fashion = CacheClassLabel(val_dataset)

    val_dataset_mnist = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    # val_dataset_mnist = CacheClassLabel(val_dataset)
    train_dataset_mnist.targets = train_dataset_mnist.targets + 10
    val_dataset_mnist.targets = val_dataset_mnist.targets + 10
    train_dataset = ConcatDataset([train_dataset_fashion, train_dataset_mnist])
    train_dataset.root = train_dataset_mnist.root
    val_dataset = ConcatDataset([val_dataset_fashion, val_dataset_mnist])
    val_dataset.root = val_dataset_mnist.root
    val_dataset = CacheClassLabel(val_dataset)
    train_dataset = CacheClassLabel(train_dataset)
    return train_dataset, val_dataset, 28, 1


def CIFAR10(dataroot, skip_normalization=False, train_aug=False):
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3


def CERN(dataroot, skip_normalization=False, train_aug=True, test_split=0.25):
    data_cond = np.load(f'{dataroot}/cern/data_nonrandom_particles.npz')["arr_0"]
    data_cond = pd.DataFrame(data_cond, columns=['Energy', 'Vx', 'Vy', 'Vz', 'Px', 'Py', 'Pz', 'mass', 'charge'])
    data = np.load(f'{dataroot}/cern/data_nonrandom_responses.npz')["arr_0"]
    n_classes = 10
    bin_labels = list(range(n_classes))
    data_cond["label"] = pd.qcut(data_cond['Energy'], q=n_classes, labels=bin_labels)
    data = np.log(data + 1)
    data = np.expand_dims(data, 1)
    train_cond = data_cond.sample(int(len(data_cond) * (1 - test_split)))
    test_cond = data_cond.drop(train_cond.index)

    train_dataset = TensorDataset(torch.Tensor(data[train_cond.index]).float(),
                                  torch.Tensor(train_cond["label"].values.astype(int)).long())
    test_dataset = TensorDataset(torch.Tensor(data[test_cond.index]).float(),
                                 torch.Tensor(test_cond["label"].values.astype(int)).long())

    train_dataset.root = dataroot
    train_dataset = CacheClassLabel(train_dataset)
    test_dataset.root = dataroot
    test_dataset = CacheClassLabel(test_dataset)
    raise NotImplementedError() #Check size
    return train_dataset, test_dataset


def Flowers(dataroot, skip_normalization=False, train_aug=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    size = 64
    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        if skip_normalization:
            train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(100),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(100),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    train_dir = dataroot + "/flower_data/train/"
    val_dir = dataroot + "/flower_data/valid/"
    # train_dir = dataroot + "/flowers_selected/"
    # val_dir = dataroot + "/flowers_selected/"
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    # If doesn't work please download data from https://www.kaggle.com/c/oxford-102-flower-pytorch

    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=train_transform)

    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 64, 3


def CIFAR100(dataroot, skip_normalization=False, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset, 32, 3

