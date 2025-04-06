import questionary
from questionary import Choice
import yaml
import os
from dataset_loader import list_builtin_datasets, load_builtin_dataset, load_user_dataset
from model_loader import get_builtin_model
from glob import glob
import torch
import random


def select_dataset():
    dataset_shapes = {
        "mnist": [1, 28, 28],
        "fashionmnist": [1, 28, 28],
        "cifar10": [3, 32, 32],
        "cifar100": [3, 32, 32],
        "svhn": [3, 32, 32],
        "kmnist": [1, 28, 28],
        "emnist": [1, 28, 28]
    }

    choices = [f"{name} (built-in)" for name in list_builtin_datasets()] + ["user_dataset.py"]
    selected = questionary.select("Select a dataset:", choices=choices).ask()

    if "user" in selected:
        dataset_info = {"type": "custom", "name": "user_dataset.py"}
        try:
            train, _, _ = load_user_dataset()
            num_classes = detect_num_classes(train)
        except Exception as e:
            print(f"[!] Could not load custom dataset: {e}")
            num_classes = None
        input_shape = [1, 28, 28]  # fallback default
    else:
        dataset_name = selected.split(" ")[0].lower()
        dataset_info = {"type": "builtin", "name": dataset_name}
        try:
            train, _, _ = load_builtin_dataset(dataset_name)
            num_classes = detect_num_classes(train)
        except Exception as e:
            print(f"[!] Could not load built-in dataset '{dataset_name}': {e}")
            num_classes = None
        input_shape = dataset_shapes.get(dataset_name, [1, 28, 28])

    if num_classes is None:
        num_classes = int(questionary.text("How many output classes does your dataset have?", default="10").ask())

    return dataset_info, num_classes, input_shape


def detect_num_classes(dataset):
    try:
        targets = [int(dataset.dataset.targets[i]) for i in dataset.indices]
        return len(set(targets))
    except:
        return None


def select_model(num_classes):
    choices = ["cnn", "mlp", "resnet18", "resnet50", "vit", "user_model.py"]
    selected = questionary.select("Select a model:", choices=choices).ask()

    model_info = {
        "type": "custom" if "user" in selected else "builtin",
        "name": selected,
        "num_classes": num_classes
    }

    if model_info["type"] == "builtin" and selected in ["cnn", "mlp"]:
        params = {}
        if selected == "cnn":
            filters = questionary.text("Number of conv filters (default: 32):", default="32").ask()
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            params = {"conv_filters": int(filters), "hidden_size": int(hidden)}

        elif selected == "mlp":
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            input_size = questionary.text("Input size (default: 784):", default="784").ask()
            params = {"hidden_size": int(hidden), "input_size": int(input_size)}

        model_info["params"] = params

    return model_info


def select_profile():
    profiles = glob("../profiles/*.yaml")
    if not profiles:
        print("No profiles found in ../profiles/")
        return None

    selected = questionary.select("Select a threat profile to use:", choices=profiles).ask()
    return selected

def suggest_data_poisoning(profile_data):
    print("\n=== Attack Parameter Suggestion: Label Flipping ===\n")
    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    num_classes = profile_data.get("model", {}).get("num_classes", 10)
    classes = list(range(num_classes))
    
    if goal == "targeted":
        if data_source == "user_generated":
            strategy = "one_to_one"
            flip_rate = 0.05
        else:
            strategy = "many_to_one"
            flip_rate = 0.08
    else:
        strategy = "fully_random"
        flip_rate = 0.1 if data_source == "external_public" else 0.08

    source_class = None
    target_class = None

    if strategy == "many_to_one":
        target_class = random.choice(classes)

    elif strategy == "one_to_one":
        source_class = random.choice(classes)
        target_class = random.choice([c for c in classes if c != source_class])

    # Mostrar sugestões
    print(f"Suggested strategy: {strategy}")
    print(f"Suggested flip_rate: {flip_rate}")
    if source_class is not None:
        print(f"Suggested source class: {source_class}")
    if target_class is not None:
        print(f"Suggested target class: {target_class}")

    confirm = questionary.confirm("Do you want to accept these suggestions?").ask()
    if not confirm:
        # Estratégia
        strategy = questionary.select(
            "Choose flipping strategy:",
            choices=[
                Choice("Fully random (random->random)", value="fully_random"),
                Choice("Random to fixed (many->one)", value="many_to_one"),
                Choice("Fixed to fixed (one->one)", value="one_to_one")
            ]
        ).ask()

        # Flip rate
        flip_rate = float(questionary.text("Flip rate (e.g., 0.08):", default=str(flip_rate)).ask())

        # Classes conforme estratégia
        if strategy == "many_to_one":
            auto_target = questionary.confirm("Pick target class randomly?").ask()
            if auto_target:
                target_class = random.choice(classes)
            else:
                target_class = int(questionary.text("Enter target class to flip TO:").ask())
            source_class = None

        elif strategy == "one_to_one":
            auto_source = questionary.confirm("Pick source class randomly?").ask()
            if auto_source:
                source_class = random.choice(classes)
            else:
                source_class = int(questionary.text("Enter source class to flip FROM:").ask())

            auto_target = questionary.confirm("Pick target class randomly?").ask()
            if auto_target:
                target_class = random.choice([c for c in classes if c != source_class])
            else:
                target_class = int(questionary.text("Enter target class to flip TO:").ask())
        else:
            source_class = None
            target_class = None

    # Salvar no perfil
    profile_data["attack_overrides"] = {
        "data_poisoning": {
            "strategy": strategy,
            "flip_rate": flip_rate,
            "source_class": source_class,
            "target_class": target_class
        }
    }

    print("\n[✔] Attack configuration saved in profile.")


def run_setup():
    print("\n=== Safe-DL Framework — Module 2 Setup Wizard ===\n")

    # Dataset and model selection
    dataset_info, num_classes = select_dataset()
    model_info = select_model(num_classes)
    profile_path = select_profile()

    if profile_path is None:
        print("[!] No profile selected. Exiting.")
        return

    # Load the existing profile
    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)

    # Replace dataset and model fields
    profile_data["dataset"] = dataset_info
    profile_data["model"] = model_info

    # Suggest attack configuration and save it
    suggest_data_poisoning(profile_data)

    with open(profile_path, "w") as f:
        yaml.dump(profile_data, f)

    print(f"\n[✔] Profile updated and saved at: {profile_path}")



if __name__ == "__main__":
    run_setup()
