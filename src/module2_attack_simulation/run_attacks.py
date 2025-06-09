import yaml
import os
import torch
import json
import questionary
from torch.utils.data import DataLoader
from dataset_loader import load_builtin_dataset, load_user_dataset
from attacks.utils import evaluate_model, save_model, load_model_cfg_from_profile, load_model


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

    clean_model = load_model_cfg_from_profile(profile)

    train_model(clean_model, trainset, valset, epochs=15, class_names=class_names)
    save_model(clean_model,profile.get("name"), "clean_model")
    baseline_acc, per_class_acc = evaluate_model(clean_model, testset, class_names=class_names)

    baseline_results = {"overall_accuracy": baseline_acc, "per_class_accuracy": per_class_acc}

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_accuracy.json", "w") as f:
        json.dump(baseline_results, f, indent=4)

def run_attacks(profile,trainset, testset, valset, class_names):

    threat_categories = profile.get("threat_model", {}).get("threat_categories", [])

    if "data_poisoning" in threat_categories:
        print("[*] Running Data Poisoning attacks...")

        dp_attacks = profile.get("attack_overrides", {}).get("data_poisoning", {})

        if "label_flipping" in dp_attacks:
            print("  - Executing Label Flipping...")
            from attacks.data_poisoning.label_flipping.run_label_flipping import run_label_flipping

            label_flipping_model = load_model_cfg_from_profile(profile)

            run_label_flipping(trainset, testset, valset, label_flipping_model, profile, class_names)

        if "clean_label" in dp_attacks:
            print("  - Executing Clean Label...")
            from attacks.data_poisoning.clean_label.run_clean_label import run_clean_label

            clean_label_model = load_model_cfg_from_profile(profile)

            run_clean_label(trainset,testset,valset,clean_label_model, profile, class_names)

    if "backdoor_attacks" in threat_categories:
        print("[*] Running Backdoor attacks...")

        backdoor_attacks = profile.get("attack_overrides", {}).get("backdoor", {})

        if "static_patch" in backdoor_attacks:
            print("  - Executing Static Patch...")
            from attacks.backdoor.static_patch.run_static_patch import run_static_patch

            static_patch_model = load_model_cfg_from_profile(profile)
            run_static_patch(trainset, testset, valset, static_patch_model, profile, class_names)

        if "learned" in backdoor_attacks:
            print("  - Executing Learned Trigger...")
            from attacks.backdoor.learned_trigger.run_learned_trigger import run_learned_trigger

            learned_trigger_model = load_model_cfg_from_profile(profile)
            run_learned_trigger(trainset, testset, valset, learned_trigger_model, profile, class_names)

    if "evasion_attacks" in threat_categories:
        print("[*] Running Evasion attacks...")

        evasion_attacks = profile.get("attack_overrides", {}).get("evasion", {})

        if "fgsm" in evasion_attacks:
            print("  - Executing FGSM...")
            from attacks.evasion.fgsm.run_fgsm import run_fgsm
            run_fgsm(testset, profile, class_names)

        if "pgd" in evasion_attacks:
            print("  - Executing PGD...")
            from attacks.evasion.pgd.run_pgd import run_pgd
            run_pgd(testset, profile, class_names)

        if "cw" in evasion_attacks:
            print("  - Executing C&W...")
            from attacks.evasion.cw.run_cw import run_cw
            run_cw(testset, profile, class_names)

        if "deepfool" in evasion_attacks:
            print("  - Executing DeepFool...")
            from attacks.evasion.deepfool.run_deepfool import run_deepfool
            run_deepfool(testset, profile, class_names)

        if "nes" in evasion_attacks:
            print("  - Executing NES...")
            from attacks.evasion.nes.run_nes import run_nes
            run_nes(testset, profile, class_names)

        if "spsa" in evasion_attacks:
            print("  - Executing SPSA...")
            from attacks.evasion.spsa.run_spsa import run_spsa
            run_spsa(testset, profile, class_names)

        if "transfer" in evasion_attacks:
            print("  - Executing Transfer...")
            from attacks.evasion.transfer.run_transfer import run_transfer
            run_transfer(trainset, testset,valset, profile, class_names)


def main():
    print("=== Safe-DL: Attack Simulation Module ===\n")
    profile_name = choose_profile()

    print("\n[*] Loading profile...")
    profile = load_profile(profile_name)

    print("[*] Loading dataset from profile...")
    trainset, testset, valset, class_names, num_classes = load_dataset_from_profile(profile)

    print("[*] Training clean model...")
    train_clean_model(profile, trainset, testset, valset, class_names)

    print("[*] Starting attack simulations...\n")
    run_attacks(profile,trainset, testset, valset, class_names)


if __name__ == "__main__":
    main()
