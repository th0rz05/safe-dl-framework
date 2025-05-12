import copy
import os
import yaml
import questionary
from questionary import Choice
from glob import glob
from defense_config.defense_tags import DEFENSE_TAGS
from defense_config.defense_descriptions import DEFENSE_DESCRIPTIONS


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

def configure_data_poisoning_defenses(profile_data):
    print("[*] Configuring defenses for Data Poisoning attacks...")
    recs = profile_data.get("risk_analysis", {}).get("recommendations", {})
    applied = {}

    for subattack in DEFENSE_TAGS["data_poisoning"]:
        suggestions = recs.get(subattack, [])
        all_possible = DEFENSE_TAGS["data_poisoning"][subattack]
        defenses = ask_defenses(subattack, suggestions, all_possible)
        if defenses:
            applied[subattack] = copy.deepcopy(defenses)

    return applied

def configure_backdoor_defenses(profile_data):
    print("[*] Configuring defenses for Backdoor attacks...")
    recs = profile_data.get("risk_analysis", {}).get("recommendations", {})
    applied = {}

    for subattack in DEFENSE_TAGS.get("backdoor", {}):
        suggestions = recs.get(subattack, [])
        all_possible = DEFENSE_TAGS["backdoor"][subattack]
        defenses = ask_defenses(subattack, suggestions, all_possible)
        if defenses:
            applied[subattack] = copy.deepcopy(defenses)

    return applied

def configure_evasion_defenses(profile_data):
    print("[*] Configuring defenses for Evasion attacks...")
    recs = profile_data.get("risk_analysis", {}).get("recommendations", {})
    applied = {}

    for subattack in DEFENSE_TAGS.get("evasion_attacks", {}):
        suggestions = recs.get(subattack, [])
        all_possible = DEFENSE_TAGS["evasion_attacks"][subattack]
        defenses = ask_defenses(subattack, suggestions, all_possible)
        if defenses:
            applied[subattack] = copy.deepcopy(defenses)

    return applied

def run_setup():
    print("\n=== Safe-DL — Module 4 Setup Wizard (Data Poisoning Only) ===\n")

    profile_path = select_profile()
    profile_data = load_profile(profile_path)

    if "defense_config" not in profile_data:
        profile_data["defense_config"] = {}

    if "data_poisoning" in DEFENSE_TAGS:
        dp_defenses = configure_data_poisoning_defenses(profile_data)
        profile_data["defense_config"]["data_poisoning"] = dp_defenses

    if "backdoor" in DEFENSE_TAGS:
        bd_defenses = configure_backdoor_defenses(profile_data)
        profile_data["defense_config"]["backdoor"] = bd_defenses

    if "evasion_attacks" in DEFENSE_TAGS:
        ev_defenses = configure_evasion_defenses(profile_data)
        profile_data["defense_config"]["evasion_attacks"] = ev_defenses

    save_profile(profile_data, profile_path)
    print(f"\n[✔] Profile updated with defenses and saved to: {profile_path}")


if __name__ == "__main__":
    run_setup()
