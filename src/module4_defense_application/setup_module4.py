import copy
import os
import yaml
import questionary
from questionary import Choice
from glob import glob
from defense_config.defense_tags import DEFENSE_TAGS
from defense_config.defense_descriptions import DEFENSE_DESCRIPTIONS
from defense_config.defense_parameters import DEFENSE_PARAMETER_FUNCTIONS


def load_profile(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_profile(profile, path):
    with open(path, "w") as f:
        yaml.safe_dump(profile, f)

def get_available_profiles():
    return glob("../profiles/*.yaml")

def select_profile():
    profiles = get_available_profiles()
    if not profiles:
        print("[!] No profiles found.")
        exit(1)
    return questionary.select("Select a threat profile:", choices=profiles).ask()

def ask_defenses(subattack, suggestions, all_possible):
    print(f"\n=== {subattack.upper()} ===")
    print(f"Suggested defenses: {suggestions}")

    accept = questionary.confirm("Accept all suggested defenses?", default=True).ask()

    if accept:
        return suggestions

    choices = [
        Choice(f"{d} — {DEFENSE_DESCRIPTIONS.get(d, 'No description available.')}", value=d)
        for d in all_possible
    ]

    selected = questionary.checkbox(
        "Select defenses to apply:",
        choices=choices
    ).ask()

    return selected

def configure_defenses(profile_data, category):
    print(f"[*] Configuring defenses for {category.replace('_', ' ').title()}...")
    recs = profile_data.get("risk_analysis", {}).get("recommendations", {})
    applied = {}

    for subattack in DEFENSE_TAGS.get(category, {}):
        suggestions = recs.get(subattack, [])
        all_possible = DEFENSE_TAGS[category][subattack]
        defenses = ask_defenses(subattack, suggestions, all_possible)

        if defenses:
            applied[subattack] = {
                "defenses": copy.deepcopy(defenses)
            }
            for defense in defenses:
                param_fn = DEFENSE_PARAMETER_FUNCTIONS.get(defense)
                if param_fn:
                    print(f"[?] Configuring parameters for defense: {defense}")
                    params = param_fn()
                    applied[subattack][defense] = copy.deepcopy(params)

    return applied

def run_setup():
    print("\n=== Safe-DL — Module 4 Setup Wizard ===\n")

    profile_path = select_profile()
    profile_data = load_profile(profile_path)

    if "defense_config" not in profile_data:
        profile_data["defense_config"] = {}

    for category in DEFENSE_TAGS:
        defenses = configure_defenses(profile_data, category)
        profile_data["defense_config"][category] = defenses

    save_profile(profile_data, profile_path)
    print(f"\n[✔] Profile updated with defenses and saved to: {profile_path}")


if __name__ == "__main__":
    run_setup()
