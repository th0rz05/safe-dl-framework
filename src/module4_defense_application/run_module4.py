import os
import sys
import yaml
import questionary

from defenses.data_cleaning.data_cleaning import run_data_cleaning_defense

# Add module2 path for shared functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "module2_attack_simulation")))
from dataset_loader import load_builtin_dataset, load_user_dataset


def choose_profile():
    profiles_path = os.path.join("../", "profiles")
    profiles = sorted([f for f in os.listdir(profiles_path) if f.endswith(".yaml")])
    if not profiles:
        raise FileNotFoundError("No .yaml files found in ../profiles")
    return questionary.select("Choose a threat profile:", choices=profiles).ask()


def load_profile(filename):
    path = os.path.join("../", "profiles", filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_from_profile(profile):
    dataset_info = profile.get("dataset", {})
    name = dataset_info.get("name")
    dataset_type = dataset_info.get("type")

    if dataset_type == "custom":
        return load_user_dataset()
    else:
        return load_builtin_dataset(name)


def apply_data_poisoning_defenses(profile, trainset, testset, valset, class_names):
    dp_defenses = profile.get("defense_config", {}).get("data_poisoning", {})
    for attack_type, config in dp_defenses.items():
        print(f"\n[*] Applying data poisoning defenses for: {attack_type}")
        for defense_name in config.get("defenses", []):
            if defense_name == "data_cleaning":
                run_data_cleaning_defense(profile, trainset, testset, valset, class_names,attack_type)
            else:
                print(f"  - Placeholder: Running {defense_name} defense for {attack_type}")


def apply_backdoor_defenses(profile, trainset, testset, valset, class_names):
    bd_defenses = profile.get("defense_config", {}).get("backdoor", {})
    for attack_type, config in bd_defenses.items():
        print(f"\n[*] Applying backdoor defenses for: {attack_type}")
        for defense_name in config.get("defenses", []):
            print(f"  - Placeholder: Running {defense_name} defense for {attack_type}")


def apply_evasion_defenses(profile, trainset, testset, valset, class_names):
    ev_defenses = profile.get("defense_config", {}).get("evasion_attacks", {})
    for attack_type, config in ev_defenses.items():
        print(f"\n[*] Applying evasion defenses for: {attack_type}")
        for defense_name in config.get("defenses", []):
            print(f"  - Placeholder: Running {defense_name} defense for {attack_type}")


def main():
    print("=== Safe-DL: Defense Application Module (Module 4) ===\n")
    profile_name = choose_profile()
    profile = load_profile(profile_name)

    print("\n[*] Loading dataset...")
    trainset, testset, valset, class_names, _ = load_dataset_from_profile(profile)

    print("[*] Starting defense application...\n")

    apply_data_poisoning_defenses(profile, trainset, testset, valset, class_names)
    apply_backdoor_defenses(profile, trainset, testset, valset, class_names)
    apply_evasion_defenses(profile, trainset, testset, valset, class_names)

    print("\n[✔] All defenses processed.")


if __name__ == "__main__":
    main()
