import yaml
import os
import torch
import json
import questionary
from torch.utils.data import DataLoader
from dataset_loader import load_builtin_dataset, load_user_dataset
from model_loader import get_builtin_model, load_user_model
from attacks.utils import evaluate_model


def choose_profile():
    profiles_path = os.path.join("../", "profiles")
    profiles = sorted([f for f in os.listdir(profiles_path) if f.endswith(".yaml")])

    if not profiles:
        raise FileNotFoundError("No .yaml files found in ../profiles")

    selected = questionary.select(
        "Choose a threat profile:",
        choices=profiles
    ).ask()

    return selected


def load_profile(filename):
    path = os.path.join("../", "profiles", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_profile(profile):
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


def load_dataset_from_profile(profile):
    dataset_info = profile.get("dataset", {})
    name = dataset_info.get("name")
    dataset_type = dataset_info.get("type")

    if dataset_type == "custom":
        return load_user_dataset()
    else:
        return load_builtin_dataset(name)

def train_clean_model(profile, trainset, testset, valset, class_names):
    print("[*] Training baseline model (clean data)...")
    from attacks.utils import train_model

    clean_model = load_model_from_profile(profile)

    train_model(clean_model, trainset, valset, epochs=3, class_names=class_names)
    baseline_acc, per_class_acc = evaluate_model(clean_model, testset, class_names=class_names)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_accuracy.json", "w") as f:
        json.dump({"accuracy": baseline_acc, "per_class_accuracy": per_class_acc}, f)


def run_attacks(profile,trainset, testset, valset, class_names):

    threat_categories = profile.get("threat_model", {}).get("threat_categories", [])

    if "data_poisoning" in threat_categories:
        print("[*] Running Data Poisoning attacks...")

        dp_attacks = profile.get("attack_overrides", {}).get("data_poisoning", {})

        if "label_flipping" in dp_attacks:
            print("  - Executing Label Flipping...")
            from attacks.data_poisoning.label_flipping.run_label_flipping import run_label_flipping

            label_flipping_model = load_model_from_profile(profile)

            run_label_flipping(trainset, testset, valset, label_flipping_model, profile, class_names)

        if "clean_label" in dp_attacks:
            print("  - Executing Clean Label...")
            from attacks.data_poisoning.clean_label.run_clean_label import run_clean_label

            clean_label_model = load_model_from_profile(profile)

            run_clean_label(trainset,testset,valset,clean_label_model, profile, class_names)


def main():
    print("=== Safe-DL: Attack Simulation Module ===\n")
    profile_name = choose_profile()

    print("\n[*] Loading profile...")
    profile = load_profile(profile_name)

    print("[*] Loading dataset from profile...")
    trainset, testset, valset, class_names, num_classes = load_dataset_from_profile(profile)

    print("[*] Training clean model...")
    #train_clean_model(profile, trainset, testset, valset, class_names)
    
    print("[*] Starting attack simulations...\n")
    run_attacks(profile,trainset, testset, valset, class_names)


if __name__ == "__main__":
    main()
