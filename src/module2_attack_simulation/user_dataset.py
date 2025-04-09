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

    # Get number of classes
    targets = [int(full_train.targets[i]) for i in range(len(full_train))]
    num_classes = len(set(targets))

    # Get class names (MNIST doesn't include them, so fallback to string numbers)
    class_names = [str(i) for i in range(num_classes)]

    return trainset, testset, valset, class_names, num_classes


# Optional test run
if __name__ == "__main__":
    train, test, val, class_names, num_classes = get_dataset()
    print("Detected labels:", class_names)
    print("Num classes:", num_classes)
