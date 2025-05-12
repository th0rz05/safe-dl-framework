import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model_loader import load_user_model, get_builtin_model

from tqdm.auto import tqdm


def train_model(model,
                trainset,
                valset=None,
                epochs=3,
                batch_size=64,
                class_names=None,
                lr=1e-3,
                silent=False):
    """
    Standard training loop with tqdm progress bars.

    Args
    ----
    model : nn.Module
    trainset / valset : torch.utils.data.Dataset
    epochs : int
    batch_size : int
    lr : float            Adam learning‑rate
    silent : bool         True → suppress progress bars (for unit tests)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size) if valset else None

    # epoch loop
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        it = train_loader if silent else tqdm(train_loader,
                                              desc=f"[Train] Epoch {ep}/{epochs}",
                                              leave=False)
        for x, y in it:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {ep:02d}: Train loss = {running_loss:.4f}")

        # optional validation
        if val_loader:
            acc, _ = evaluate_model(model,
                                    valset,
                                    class_names=class_names,
                                    silent=silent,
                                    prefix=f"[Val]   Epoch {ep}/{epochs}")
            print(f"Epoch {ep:02d}: Validation accuracy = {acc:.4f}")

def evaluate_model(model,
                   dataset,
                   class_names=None,
                   batch_size=64,
                   silent=False,
                   prefix="[Eval]"):
    """
    Evaluate accuracy + per‑class accuracy with tqdm.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size)

    total = correct = 0
    class_correct = {}
    class_total   = {}

    iterator = loader if silent else tqdm(loader, desc=prefix, leave=False)

    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)

            preds = model(x).argmax(1)

            total   += y.size(0)
            correct += (preds == y).sum().item()

            for label, pred in zip(y, preds):
                label = int(label)
                class_total[label]   = class_total.get(label, 0) + 1
                if pred == label:
                    class_correct[label] = class_correct.get(label, 0) + 1

    overall_acc = correct / total
    print(f"{prefix} Overall accuracy: {overall_acc:.4f}")

    per_class_acc = {}
    for cls in sorted(class_total):
        tot = class_total[cls]
        cor = class_correct.get(cls, 0)
        acc = cor / tot if tot else 0.0
        name = class_names[cls] if class_names else str(cls)
        per_class_acc[name] = round(acc, 4)
        print(f"  - {name:>10} : {acc:.4f} ({cor}/{tot})")

    return overall_acc, per_class_acc




def get_class_labels(dataset):
    
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        targets = dataset.dataset.targets
    elif hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        raise ValueError("Dataset does not expose `.targets` attribute.")

    unique_labels = sorted(set(int(label) for label in targets))
    return unique_labels

def save_model(model, profile_name, model_name):
    os.makedirs("../saved_models", exist_ok=True)
    model_path = os.path.join("saved_models", f"{profile_name}_{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[✔] Model saved to {model_path}")


def load_model(model_name, profile):
    profile_name = profile.get("name", "default")
    model_path = os.path.join("saved_models", f"{profile_name}_{model_name}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[!] Model file not found at {model_path}. Make sure it was trained and saved.")

    # Load the correct model architecture based on the profile
    model = load_model_cfg_from_profile(profile)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"[✔] Loaded model from {model_path}")

    return model

def load_model_cfg_from_profile(profile):
    model_cfg = profile.get("model", {})
    model_type = model_cfg.get("type", "builtin")
    model_name = model_cfg.get("name")
    num_classes = model_cfg.get("num_classes", 10)
    input_shape = tuple(model_cfg.get("input_shape", [1, 28, 28]))
    params = model_cfg.get("params", {})

    if model_type == "custom":
        return load_user_model("user_model.py")

    elif model_type == "builtin":
        return get_builtin_model(
            name=model_name,
            num_classes=num_classes,
            input_shape=input_shape,
            **params
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
