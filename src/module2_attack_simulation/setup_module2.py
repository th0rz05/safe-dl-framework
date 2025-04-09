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
            _, _, _,class_names, num_classes = load_user_dataset()
        except Exception as e:
            print(f"[!] Could not load custom dataset: {e}")
            num_classes = None
        input_shape = [1, 28, 28]  # fallback default
    else:
        dataset_name = selected.split(" ")[0].lower()
        dataset_info = {"type": "builtin", "name": dataset_name}
        try:
            _, _, _, class_names, num_classes  = load_builtin_dataset(dataset_name)
        except Exception as e:
            print(f"[!] Could not load built-in dataset '{dataset_name}': {e}")
            num_classes = None
        input_shape = dataset_shapes.get(dataset_name, [1, 28, 28])

    if num_classes is None:
        num_classes = int(questionary.text("How many output classes does your dataset have?", default="10").ask())
        
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
        

    return dataset_info, num_classes, input_shape, class_names


def select_model(num_classes, input_shape):
    choices = ["cnn", "mlp", "resnet18", "resnet50", "vit", "user_model.py"]
    selected = questionary.select("Select a model:", choices=choices).ask()

    model_info = {
        "type": "custom" if "user" in selected else "builtin",
        "name": selected,
        "num_classes": num_classes
    }

    # Handle input shape for built-in models
    if model_info["type"] == "builtin":
        if questionary.confirm("Do you want to customize the input shape?", default=False).ask():
            channels = int(questionary.text("Number of input channels:", default=str(input_shape[0])).ask())
            height = int(questionary.text("Image height:", default=str(input_shape[1])).ask())
            width = int(questionary.text("Image width:", default=str(input_shape[2])).ask())
            model_info["input_shape"] = [channels, height, width]
        else:
            model_info["input_shape"] = input_shape

        # Optional parameters for CNN and MLP
        if selected == "cnn":
            filters = questionary.text("Number of conv filters (default: 32):", default="32").ask()
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            model_info["params"] = {
                "conv_filters": int(filters),
                "hidden_size": int(hidden)
            }

        elif selected == "mlp":
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            input_size = questionary.text("Input size (default: 784):", default="784").ask()
            model_info["params"] = {
                "hidden_size": int(hidden),
                "input_size": int(input_size)
            }

    return model_info



def select_profile():
    profiles = glob("../profiles/*.yaml")
    if not profiles:
        print("No profiles found in ../profiles/")
        return None

    selected = questionary.select("Select a threat profile to use:", choices=profiles).ask()
    return selected

def configure_label_flipping(profile_data, class_names):
    print("\n=== Attack Parameter Suggestion: Label Flipping ===\n")
    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    num_classes = profile_data.get("model", {}).get("num_classes", 10)
    classes = list(range(num_classes))

    # Default suggestion
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

    print(f"Suggested strategy: {strategy}")
    print(f"Suggested flip_rate: {flip_rate}")
    if source_class is not None:
        print(f"Suggested source class: {source_class} – {class_names[source_class]}")
    if target_class is not None:
        print(f"Suggested target class: {target_class} – {class_names[target_class]}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        strategy = questionary.select(
            "Choose flipping strategy:",
            choices=[
                Choice("Fully random (random→random)", value="fully_random"),
                Choice("Random to fixed (many→one)", value="many_to_one"),
                Choice("Fixed to fixed (one→one)", value="one_to_one")
            ]
        ).ask()

        flip_rate = float(questionary.text("Flip rate (e.g., 0.08):", default=str(flip_rate)).ask())

        class_options = [f"{i} – {name}" for i, name in enumerate(class_names)]

        if strategy == "many_to_one":
            if questionary.confirm("Pick target class randomly?").ask():
                target_class = random.choice(classes)
            else:
                target_class_str = questionary.select("Select target class (flip TO):", choices=class_options).ask()
                target_class = int(target_class_str.split(" ")[0])
            source_class = None

        elif strategy == "one_to_one":
            if questionary.confirm("Pick source class randomly?").ask():
                source_class = random.choice(classes)
            else:
                source_class_str = questionary.select("Select source class (flip FROM):", choices=class_options).ask()
                source_class = int(source_class_str.split(" ")[0])

            if questionary.confirm("Pick target class randomly?").ask():
                target_class = random.choice([c for c in classes if c != source_class])
            else:
                target_class_str = questionary.select("Select target class (flip TO):", choices=class_options).ask()
                target_class = int(target_class_str.split(" ")[0])
        else:
            source_class = None
            target_class = None

    cfg = {
                "strategy": strategy,
                "flip_rate": flip_rate,
                "source_class": source_class,
                "target_class": target_class
            }

    return cfg

def run_setup():
    print("\n=== Safe-DL Framework — Module 2 Setup Wizard ===\n")

    dataset_info, num_classes, input_shape, class_names = select_dataset()
    model_info = select_model(num_classes, input_shape)
    profile_path = select_profile()

    if profile_path is None:
        print("[!] No profile selected. Exiting.")
        return

    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)

    profile_data["dataset"] = dataset_info
    profile_data["model"] = model_info

    threat_categories = profile_data.get("threat_model", {}).get("threat_categories", [])

    if "data_poisoning" not in threat_categories:
        print("[*] Data poisoning not selected in threat model. Skipping...")
    else:
        selected_attacks = questionary.checkbox(
            "Select the data poisoning attacks to simulate:",
            choices=[
                Choice("Label Flipping", value="label_flipping"),
                Choice("Clean Label ", value="clean_label"),
            ]
        ).ask()

        data_poisoning_cfg = {}

        if "label_flipping" in selected_attacks:
            data_poisoning_cfg["label_flipping"] = configure_label_flipping(profile_data,class_names)

        profile_data["attack_overrides"] = profile_data.get("attack_overrides", {})
        profile_data["attack_overrides"]["data_poisoning"] = data_poisoning_cfg

    with open(profile_path, "w") as f:
        yaml.dump(profile_data, f)

    print(f"\n[✔] Profile updated and saved at: {profile_path}")


if __name__ == "__main__":
    run_setup()
