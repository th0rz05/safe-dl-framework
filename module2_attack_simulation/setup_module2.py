import questionary
import yaml
import os
from dataset_loader import list_builtin_datasets
from model_loader import get_builtin_model
from glob import glob


def select_dataset():
    choices = [f"{name} (built-in)" for name in list_builtin_datasets()] + ["user_dataset.py"]
    selected = questionary.select("Select a dataset:", choices=choices).ask()

    if "user" in selected:
        return {"type": "custom", "name": "user_dataset.py"}
    else:
        return {"type": "builtin", "name": selected.split(" ")[0]}


def select_model():
    choices = ["cnn", "mlp", "resnet18", "resnet50", "vit", "user_model.py"]
    selected = questionary.select("Select a model:", choices=choices).ask()

    model_info = {
        "type": "custom" if "user" in selected else "builtin",
        "name": selected
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


def run_setup():
    print("\n=== Safe-DL Framework — Module 2 Setup Wizard ===\n")

    dataset_info = select_dataset()
    model_info = select_model()
    profile_path = select_profile()

    print("\nConfiguration complete!\n")

    # Ask to save this as a new profile or overwrite selected one
    save_new = questionary.confirm("Do you want to save a new combined profile with dataset and model info?").ask()

    profile_data = {}
    if profile_path:
        with open(profile_path, "r") as f:
            profile_data = yaml.safe_load(f)

    # Update or add model/dataset sections
    profile_data["dataset"] = dataset_info
    profile_data["model"] = model_info

    if save_new:
        filename = questionary.text("Enter filename (e.g., my_combined_profile.yaml):").ask()
        final_path = os.path.join("../profiles", filename)
    else:
        final_path = profile_path

    with open(final_path, "w") as f:
        yaml.dump(profile_data, f)

    print(f"\n[✔] Final profile saved at: {final_path}")


if __name__ == "__main__":
    run_setup()
