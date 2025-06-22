import os
from torchvision import datasets, transforms
from torch.utils.data import random_split
import importlib.util
import torch


def load_builtin_dataset(name, augment=False):
    """
    Load built-in dataset with optional augmentation (only applied to training set).
    """

    mean, std = get_normalization_params("cifar10")

    transform_augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_static = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform = transform_augmented if augment else transform_static

    if name == "mnist":
        full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_static)

    elif name == "fashionmnist":
        full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_static)

    elif name == "kmnist":
        full_train = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform_static)

    elif name == "cifar10":
        full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_static)

    elif name == "cifar100":
        full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_static)

    elif name == "svhn":
        full_train = datasets.SVHN(root="./data", split='train', download=True, transform=transform)
        testset = datasets.SVHN(root="./data", split='test', download=True, transform=transform_static)

    elif name == "emnist":
        full_train = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
        testset = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform_static)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split train into train + val (90/10)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    trainset, valset = random_split(full_train, [train_size, val_size])

    num_classes = detect_num_classes(trainset)
    class_names = get_class_names(trainset, num_classes)

    return trainset, testset, valset, class_names, num_classes

def get_normalization_params(name):
    """
    Devolve os parâmetros de normalização (mean, std) para um dataset.
    """
    normalize = {
        "mean": {
            "mnist": [0.1307],
            "fashionmnist": [0.2860],
            "kmnist": [0.1904],
            "cifar10": [0.4914, 0.4822, 0.4465],
            "cifar100": [0.5071, 0.4865, 0.4409],
            "svhn": [0.4377, 0.4438, 0.4728],
            "emnist": [0.1751]
        },
        "std": {
            "mnist": [0.3081],
            "fashionmnist": [0.3530],
            "kmnist": [0.3475],
            "cifar10": [0.2023, 0.1994, 0.2010],
            "cifar100": [0.2673, 0.2564, 0.2761],
            "svhn": [0.1980, 0.2010, 0.1970],
            "emnist": [0.3333]
        }
    }

    if name not in normalize["mean"]:
        raise ValueError(f"Unknown dataset '{name}' for normalization params")

    return normalize["mean"][name], normalize["std"][name]

def unnormalize(img, mean, std):
    """
    Reverte a normalização de um tensor de imagem.
    Suporta tensor de shape [C, H, W] ou [B, C, H, W].
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if img.dim() == 4:  # batch [B, C, H, W]
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return img * std + mean


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

def detect_num_classes(dataset):
    try:
        targets = [int(dataset.dataset.targets[i]) for i in dataset.indices]
        return len(set(targets))
    except:
        return None
    
def get_class_names(dataset,num_classes):

    if num_classes is None:
        return None
    
    try:
        class_names = dataset.dataset.classes
    except AttributeError:
        class_names = [str(i) for i in range(num_classes)]
        
    return class_names