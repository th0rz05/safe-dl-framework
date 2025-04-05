import yaml
import importlib
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import questionary

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

def load_model():
    user_model = importlib.import_module("user_model")
    return user_model.get_model()

def load_dataset():
    user_dataset = importlib.import_module("user_dataset")
    return user_dataset.get_dataset()

def evaluate(model, dataset, desc=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=64)
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
    # Baseline: t´reinar com dataset limpo
    print("[*] Training baseline model (clean data)...")
    from attacks.utils import train_model

    clean_model = model.__class__()  # clone model
    train_model(clean_model, trainset, valset, epochs=3)
    baseline_acc = evaluate(clean_model, testset, desc="clean test set")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_accuracy.json", "w") as f:
        json.dump({"baseline_accuracy": baseline_acc}, f, indent=2)

    # Ataques com base no profile
    threat_categories = profile.get("threat_model", {}).get("threat_categories", [])
    
    if "data_poisoning" in threat_categories:
        print("[*] Running Data Poisoning attack...")
        attack = importlib.import_module("attacks.data_poisoning.run")
        attack.run(trainset, testset, valset, model, profile)
        

def main():
    print("=== Safe-DL: Attack Simulation Module ===\n")
    profile_name = choose_profile()
    
    print("\n[*] Loading profile...")
    profile = load_profile(profile_name)

    print("[*] Loading user model and dataset...")
    model = load_model()
    trainset, testset, valset = load_dataset()

    print("[*] Starting attack simulations...\n")
    run_attacks(profile, model, trainset, testset, valset)

if __name__ == "__main__":
    main()
