import torch.nn as nn
import torchvision.models as models
import importlib.util
import os


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4608 if input_channels == 1 else 5408, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def get_builtin_model(name="cnn", num_classes=10, input_shape=(1, 28, 28)):
    input_channels, height, width = input_shape

    if name == "cnn":
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes)

    elif name == "mlp":
        return MLP(input_size=input_channels * height * width, num_classes=num_classes)

    elif name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "vit":
        model = models.vit_b_16(pretrained=False)
        model.heads = nn.Linear(model.heads.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model: {name}")


def load_user_model(module_path="user_model.py"):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"{module_path} not found. Please provide your own model loader.")

    spec = importlib.util.spec_from_file_location("user_model", module_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    return user_module.get_model()
