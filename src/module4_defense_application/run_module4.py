import os
import sys
import yaml
import questionary

from defenses.data_cleaning.data_cleaning import run_data_cleaning_defense
from defenses.per_class_monitoring.per_class_monitoring import run_per_class_monitoring_defense
from defenses.robust_loss.robust_loss import run_robust_loss_defense
from defenses.dp_training.dp_training import run_dp_training_defense
from defenses.provenance_tracking.provenance_tracking import run_provenance_tracking_defense
from defenses.influence_functions.influence_functions import run_influence_functions_defense
from defenses.activation_clustering.activation_clustering import run_activation_clustering_defense
from defenses.spectral_signatures.spectral_signatures import run_spectral_signatures_defense
from defenses.anomaly_detection.anomaly_detection import run_anomaly_detection_defense
from defenses.pruning.pruning import run_pruning_defense
from defenses.fine_pruning.fine_pruning import run_fine_pruning_defense
from defenses.model_inspection.model_inspection import run_model_inspection_defense
from defenses.adversarial_training.adversarial_training import run_adversarial_training_defense
from defenses.randomized_smoothing.randomized_smoothing import run_randomized_smoothing_defense
from defenses.certified_defense.certified_defense import run_certified_defense



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
            elif defense_name == "per_class_monitoring":
                run_per_class_monitoring_defense(profile,attack_type)
            elif defense_name == "robust_loss":
                run_robust_loss_defense(profile,trainset,testset,valset,class_names,attack_type)
            elif defense_name == "dp_training":
                run_dp_training_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "provenance_tracking":
                run_provenance_tracking_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "influence_functions":
                run_influence_functions_defense(profile, trainset, testset, valset, class_names, attack_type)
            else:
                print(f"  - Error: Unknown defense '{defense_name}' for {attack_type}. Please check the profile configuration.")


def apply_backdoor_defenses(profile, trainset, testset, valset, class_names):
    bd_defenses = profile.get("defense_config", {}).get("backdoor", {})
    for attack_type, config in bd_defenses.items():
        print(f"\n[*] Applying backdoor defenses for: {attack_type}")
        for defense_name in config.get("defenses", []):
            if defense_name == "activation_clustering":
                run_activation_clustering_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "spectral_signatures":
                run_spectral_signatures_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "anomaly_detection":
                run_anomaly_detection_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "pruning":
                run_pruning_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "fine_pruning":
                run_fine_pruning_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "model_inspection":
                run_model_inspection_defense(profile, trainset, testset, valset, class_names, attack_type)
            else:
                print(f"  - Error: Unknown defense '{defense_name}' for {attack_type}. Please check the profile configuration.")


def apply_evasion_defenses(profile, trainset, testset, valset, class_names):
    ev_defenses = profile.get("defense_config", {}).get("evasion_attacks", {})
    for attack_type, config in ev_defenses.items():
        print(f"\n[*] Applying evasion defenses for: {attack_type}")
        for defense_name in config.get("defenses", []):
            if defense_name == "adversarial_training":
                run_adversarial_training_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "randomized_smoothing":
                run_randomized_smoothing_defense(profile, trainset, testset, valset, class_names, attack_type)
            elif defense_name == "certified_defense":
                run_certified_defense(profile, trainset, testset, valset, class_names, attack_type)
            else:
                print(f"  - Error: Unknown defense '{defense_name}' for {attack_type}. Please check the profile configuration.")


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
