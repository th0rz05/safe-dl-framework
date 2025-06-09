import os
import yaml
import json
from glob import glob
import questionary

RESULTS_BASE = "../module2_attack_simulation/results"
DEFENSE_RESULTS_BASE = "../module4_defense_application/results"

def select_profile():
    profiles = glob("../profiles/*.yaml")
    if not profiles:
        print("No profiles found in ../profiles/")
        return None

    selected = questionary.select("Select a threat profile to use:", choices=profiles).ask()
    return selected

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    profile_path = select_profile()
    if profile_path is None:
        return

    profile = load_yaml(profile_path)
    attack_overrides = profile.get("attack_overrides", {})
    defense_config = profile.get("defense_config", {})

    for attack_category, attacks in attack_overrides.items():
        print(f"\n[+] Attack Category: {attack_category}")

        for attack_name, attack_params in attacks.items():
            print(f"  - Attack: {attack_name}")
            attack_path = os.path.join(RESULTS_BASE, attack_category, attack_name)

            baseline_path = os.path.join(attack_path, "baseline_metrics.json")
            attack_path_json = os.path.join(attack_path, "attack_metrics.json")

            print(f"    > Baseline: {baseline_path}")
            print(f"    > Attack:   {attack_path_json}")

            if not os.path.exists(baseline_path) or not os.path.exists(attack_path_json):
                print("    [!] Missing metrics, skipping.")
                continue

            defenses_applied = defense_config.get(attack_category, {}).get(attack_name, [])
            print(f"    > Defenses applied: {defenses_applied}")

            for defense in defenses_applied:
                defense_json_path = os.path.join(DEFENSE_RESULTS_BASE, attack_category, attack_name, defense, "defense_results.json")
                print(f"      - Defense JSON: {defense_json_path}")

                if not os.path.exists(defense_json_path):
                    print("        [!] Missing defense results, skipping.")

if __name__ == "__main__":
    main()
