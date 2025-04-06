import yaml
import os
import torch
import json
import questionary
from torch.utils.data import DataLoader
from dataset_loader import load_builtin_dataset, load_user_dataset
from model_loader import get_builtin_model, load_user_model


def choose_profile():
    profiles_path = os.path.join("..", "profiles")
    profiles = sorted([f for f in os.listdir(profiles_path) if f.endswith(".yaml")])

    if not profiles:
        raise FileNotFoundError("No .yaml files found in ../profiles")

    selected = questionary.select(
        "Choose a threat profile:",
        choices=profiles
    ).ask()

    return selected


def load_profile(filename):
    path = os.path.join("..", "profiles", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_profile(profile):
    model_info = profile.get("model", {})
    name = model_info.get("name")
    model_type = model_info.get("type")
    num_classes = model_info.get("num_classes", 10)
    params = model_info.get("params", {})

    if model_type == "custom":
        return load_user_model()
    else:
        return get_builtin_model(name=name, num_classes=num_classes, input_shape=(1, 28, 28), **params)


def load_dataset_from_profile(profile):
    dataset_info = profile.get("dataset", {})
    name = dataset_info.get("name")
    dataset_type = dataset_info.get("type")

    if dataset_type == "custom":
        return load_user_dataset()
    else:
        return load_builtin_dataset(name)


def evaluate(model, dataset, desc=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=64)
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[+] Accuracy on {desc}: {acc:.4f}")
    return acc


def run_attacks(profile, model, trainset, testset, valset):
    print("[*] Training baseline model (clean data)...")
    from attacks.utils import train_model

    clean_model = model.__class__()  # clone model
    train_model(clean_model, trainset, valset, epochs=3)
    baseline_acc = evaluate(clean_model, testset, desc="clean test set")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_accuracy.json", "w") as f:
        json.dump({"baseline_accuracy": baseline_acc}, f, indent=2)

    # Ataques com base no perfil
    threat_categories = profile.get("threat_model", {}).get("threat_categories", [])

    if "data_poisoning" in threat_categories:
        print("[*] Running Data Poisoning attack...")
        attack = __import__("attacks.data_poisoning.run", fromlist=["run"])
        attack.run(trainset, testset, valset, model, profile)


def main():
    print("=== Safe-DL: Attack Simulation Module ===\n")
    profile_name = choose_profile()

    print("\n[*] Loading profile...")
    profile = load_profile(profile_name)

    print("[*] Loading model and dataset from profile...")
    model = load_model_from_profile(profile)
    trainset, testset, valset = load_dataset_from_profile(profile)

    print("[*] Starting attack simulations...\n")
    run_attacks(profile, model, trainset, testset, valset)


if __name__ == "__main__":
    main()
