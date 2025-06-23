import os
import yaml
import json
from glob import glob
import questionary
import importlib
from defense_score_utils import (
    evaluate_backdoor_defense,
    evaluate_data_poisoning_defense,
    evaluate_evasion_defense
)

from generate_defense_evaluation_report import generate_report



RESULTS_BASE = "../module2_attack_simulation/results"
DEFENSE_RESULTS_BASE = "../module4_defense_application/results"
RESULTS_JSON_PATH = os.path.join("results", "defense_evaluation.json")

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

    baseline_path = os.path.join(RESULTS_BASE, "baseline_accuracy.json")
    if not os.path.exists(baseline_path):
        print(f"[!] Baseline file not found at {baseline_path}")
        return

    baseline_data = load_json(baseline_path)
    print(f"[+] Baseline Accuracy: {baseline_data['overall_accuracy']}")

    all_scores = {}

    for attack_category, attacks in attack_overrides.items():
        print(f"\n[+] Attack Category: {attack_category}")
        all_scores[attack_category] = {}

        for attack_name in attacks:
            print(f"  - Attack: {attack_name}")
            attack_path = os.path.join(RESULTS_BASE, attack_category, attack_name)
            attack_metrics_path = os.path.join(attack_path, f"{attack_name}_metrics.json")

            if not os.path.exists(attack_metrics_path):
                print("    [!] Missing attack metrics, skipping.")
                continue

            attack_data = load_json(attack_metrics_path)

            config = defense_config.get(attack_category, {}).get(attack_name, {})
            defenses_applied = config.get("defenses", [])
            print(f"    > Defenses applied: {defenses_applied}")

            all_scores[attack_category][attack_name] = {}

            for defense in defenses_applied:
                defense_path = os.path.join(
                    DEFENSE_RESULTS_BASE,
                    attack_category,
                    attack_name,
                    f"{defense}_results.json"
                )

                print(f"      - Checking: {defense_path}")
                if not os.path.exists(defense_path):
                    print("        [!] Missing defense results, skipping.")
                    continue

                defense_data = load_json(defense_path)

                try:
                    if attack_category == "backdoor":
                        score = evaluate_backdoor_defense(defense, defense_data, attack_data)
                    elif attack_category == "data_poisoning":
                        score = evaluate_data_poisoning_defense(defense, defense_data, attack_data, baseline_data)
                    elif attack_category == "evasion":
                        score = evaluate_evasion_defense(defense, defense_data, attack_data,
                                                         baseline_data)  # Note: add baseline_data here too if needed for evasion

                    # Only proceed if score was successfully generated
                    if score:  # Check if score is not None or empty if your functions can return that
                        all_scores[attack_category][attack_name][defense] = score
                        print(f"      [✓] Scores for {defense}:")
                        print(f"         Mitigation Score : {score['mitigation_score']}")
                        print(f"         CAD Score        : {score['cad_score']}")
                        print(f"         Cost Score       : {score['defense_cost_score']}")
                        print(f"         Final Score      : {score['final_score']}")
                    else:
                        print(f"      [!] Evaluation function for {defense} returned empty or invalid score.")

                except Exception as e:
                    print(f"      [!] Error evaluating {defense}: {e}")

    # Save all scores to a single file
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f_out:
        json.dump(all_scores, f_out, indent=2)

    print(f"\n[✓] All scores saved to {RESULTS_JSON_PATH}")

    generate_report(
        profile_path="../profiles/" + profile['name']+ ".yaml",
        json_path="results/defense_evaluation.json",
        md_path="results/defense_evaluation_report.md"
    )


if __name__ == "__main__":
    main()
