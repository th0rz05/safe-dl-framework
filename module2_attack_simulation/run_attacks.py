import yaml
import importlib
import os

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

def run_attacks(profile, model, trainset, testset):
    threat_categories = profile.get("threat_model", {}).get("threat_categories", [])
    
    if "data_poisoning" in threat_categories:
        print("[*] Running Data Poisoning attack...")
        data_poisoning = importlib.import_module("attacks.data_poisoning.run")
        data_poisoning.run(trainset, testset, model, profile)

    # Future attacks (backdoor, adversarial, etc.)
    # if "backdoor_attacks" in threat_categories:
    #     ...

def main():
    print("=== Safe-DL: Attack Simulation Module ===\n")
    profile_name = input("Enter the name of your profile YAML file (e.g., profile.yaml): ").strip()
    
    print("\n[*] Loading profile...")
    profile = load_profile(profile_name)

    print("[*] Loading user model and dataset...")
    model = load_model()
    trainset, testset = load_dataset()

    print("[*] Starting attacks based on threat profile...\n")
    run_attacks(profile, model, trainset, testset)

if __name__ == "__main__":
    main()
