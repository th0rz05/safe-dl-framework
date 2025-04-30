import os
import json
import yaml
import questionary
from glob import glob
from pathlib import Path
from risk_utils import analyze_label_flipping, analyze_clean_label
from risk_utils import analyze_static_patch, analyze_learned_trigger

# Paths
BASELINE_PATH = Path("../module2_attack_simulation/results/baseline_accuracy.json")
RESULTS_DIR = Path("../module2_attack_simulation/results/")
OUTPUT_DIR = Path("results/risk_analysis/")
DP_DIR = RESULTS_DIR / "data_poisoning"
BD_DIR = RESULTS_DIR / "backdoor"

def select_profile():
    profiles = glob("../profiles/*.yaml")
    if not profiles:
        print("No profiles found in ../profiles/")
        return None

    selected = questionary.select("Select a threat profile to use:", choices=profiles).ask()
    return selected

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    # Select the profile file
    profile_path = select_profile()
    if profile_path is None:
        print("[!] No profile selected. Exiting.")
        return

    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)

    if not BASELINE_PATH.exists():
        print(f"[!] Baseline file not found at {BASELINE_PATH}")
        return

    baseline = load_json(BASELINE_PATH)
    analysis = {}


    attack_overrides = profile_data.get("attack_overrides", {})

    # Get list of data poisoning attacks used in the profile
    dp_attacks = attack_overrides.get("data_poisoning", {})

    # Label Flipping analysis
    if "label_flipping" in dp_attacks:
        path = DP_DIR / "label_flipping" / "label_flipping_metrics.json"
        if path.exists():
            print("[*] Analyzing attack: Label Flipping")
            data = load_json(path)
            analysis["label_flipping"] = analyze_label_flipping(data, baseline)

    # Clean Label analysis
    if "clean_label" in dp_attacks:
        path = DP_DIR / "clean_label" / "clean_label_metrics.json"
        if path.exists():
            print("[*] Analyzing attack: Clean Label")
            data = load_json(path)
            analysis["clean_label"] = analyze_clean_label(data, baseline)

    # Check backdoor attacks from profile
    bd_attacks = attack_overrides.get("backdoor", {})

    if "static_patch" in bd_attacks:
        path = BD_DIR / "static_patch" / "static_patch_metrics.json"
        if path.exists():
            print("[*] Analyzing attack: Static Patch")
            data = load_json(path)
            analysis["static_patch"] = analyze_static_patch(data)

    if "learned_trigger" in bd_attacks:
        path = BD_DIR / "learned_trigger" / "learned_trigger_metrics.json"
        if path.exists():
            print("[*] Analyzing attack: Learned Trigger")
            data = load_json(path)
            analysis["learned_trigger"] = analyze_learned_trigger(data)

    # Save the risk analysis result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "risk_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n[✓] Risk analysis saved to {output_path}")

if __name__ == "__main__":
    main()
