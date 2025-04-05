import os
from torchvision import datasets, transforms
from torch.utils.data import random_split
import importlib.util


def load_builtin_dataset(name):
    transform = transforms.ToTensor()

    if name == "mnist":
        full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "fashionmnist":
        full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "kmnist":
        full_train = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif name == "cifar100":
        full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    elif name == "svhn":
        full_train = datasets.SVHN(root="./data", split='train', download=True, transform=transform)
        testset = datasets.SVHN(root="./data", split='test', download=True, transform=transform)

    elif name == "emnist":
        full_train = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
        testset = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split train into train + val (90/10)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    trainset, valset = random_split(full_train, [train_size, val_size])

    return trainset, testset, valset


def load_user_dataset(module_path="user_dataset.py"):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"{module_path} not found. Please provide your own dataset loader.")
    
    spec = importlib.util.spec_from_file_location("user_dataset", module_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    return user_module.get_dataset()


def list_builtin_datasets():
    return [
        "mnist",
        "fashionmnist",
        "kmnist",
        "cifar10",
        "cifar100",
        "svhn",
        "emnist"
    ]
