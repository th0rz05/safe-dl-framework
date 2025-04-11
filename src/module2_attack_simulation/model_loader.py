import torch.nn as nn
import torchvision.models as models
import importlib.util
import os
import torch

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, conv_filters=32, hidden_size=128, num_classes=10, input_height=28, input_width=28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv(dummy_input)
            self.flatten_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def features(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        return self.fc[1:](x)


class MLP(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_size=128, num_classes=10):
        super().__init__()
        input_channels, height, width = input_shape
        input_size = input_channels * height * width

        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def features(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        return self.relu(x)

    def forward(self, x):
        x = self.features(x)
        return self.output(x)


def get_builtin_model(name="cnn", num_classes=10, input_shape=(1, 28, 28), **params):
    input_channels, height, width = input_shape

    if name == "cnn":
        return SimpleCNN(
            input_channels=input_channels,
            conv_filters=params.get("conv_filters", 32),
            hidden_size=params.get("hidden_size", 128),
            num_classes=num_classes,
            input_height=height,
            input_width=width
        )

    elif name == "mlp":
        return MLP(
            input_shape=input_shape,
            hidden_size=params.get("hidden_size", 128),
            num_classes=num_classes
        )

    elif name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "vit":
        model = models.vit_b_16(weights=None)
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
