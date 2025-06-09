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
    return questionary.select("Select a threat profile to use:", choices=profiles).ask()

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

    baseline_path = os.path.join(RESULTS_BASE,"baseline_accuracy.json")
    if not os.path.exists(baseline_path):
        print(f"[!] Baseline file not found at {baseline_path}")
        return

    baseline = load_json(baseline_path)
    print(f"[+] Baseline Accuracy: {baseline['overall_accuracy']}")


    for attack_category, attacks in attack_overrides.items():
        print(f"\n[+] Attack Category: {attack_category}")

        for attack_name in attacks:
            print(f"  - Attack: {attack_name}")
            attack_path = os.path.join(RESULTS_BASE, attack_category, attack_name)

            attack_metrics_path = os.path.join(attack_path,attack_name + "_metrics.json")

            print(f"    > Attack:   {attack_metrics_path}")

            if not os.path.exists(baseline_path) or not os.path.exists(attack_metrics_path):
                print("    [!] Missing metrics, skipping.")
                continue


            config = defense_config.get(attack_category, {}).get(attack_name, {})
            defenses_applied = config.get("defenses", [])

            print(f"    > Defenses applied: {defenses_applied}")

            for defense in defenses_applied:
                json_path = os.path.join(
                    DEFENSE_RESULTS_BASE,
                    attack_category,
                    attack_name,
                    f"{defense}_results.json"  # <- correto!
                )
                print(f"      - Checking: {json_path}")
                if not os.path.exists(json_path):
                    print("        [!] Missing defense results, skipping.")

if __name__ == "__main__":
    main()
