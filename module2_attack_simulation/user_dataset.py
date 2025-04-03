def get_dataset():
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    return trainset, testset
