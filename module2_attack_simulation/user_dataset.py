from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split

def get_dataset():
    transform = transforms.ToTensor()

    full_train = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)

    # Split 10% for validation
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    trainset, valset = random_split(full_train, [train_size, val_size])

    return trainset, testset, valset

if __name__ == "__main__":
    train, test, val = get_dataset()
    labels = sorted(set(int(train.dataset.targets[i]) for i in train.indices))
    print("Train labels:", labels)

