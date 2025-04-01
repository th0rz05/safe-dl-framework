import questionary
import yaml

def run_questionnaire():
    print("=== Threat Modeling Questionnaire ===\n")

    # 1. model_access
    model_access = questionary.select(
        "What level of access might an attacker have to the model?",
        choices=["white-box", "gray-box", "black-box"]
    ).ask()

    # 2. attack_goal
    attack_goal = questionary.select(
        "What is the likely goal of the attacker?",
        choices=["targeted", "untargeted"]
    ).ask()

    # 3. deployment_scenario
    deployment_scenario = questionary.select(
        "Where will the model be deployed?",
        choices=["cloud", "edge", "mobile", "api_public", "on_device"]
    ).ask()

    # 4. data_sensitivity
    data_sensitivity = questionary.select(
        "How sensitive is the training data?",
        choices=["high", "medium", "low"]
    ).ask()

    # 5. training_data_source
    training_data_source = questionary.select(
        "Where does the training data come from?",
        choices=["internal_clean", "external_public", "user_generated", "mixed"]
    ).ask()

    # 6. model_type
    model_type = questionary.select(
        "What type of architecture will the model use?",
        choices=["cnn", "transformer", "mlp", "other"]
    ).ask()

    # 7. interface_exposed
    interface_exposed = questionary.select(
        "How will the model be accessed by end users or systems?",
        choices=["api", "local_app", "sdk", "none"]
    ).ask()

    # 8. threat_categories (manual for now – dynamic suggestion comes next)
    threat_categories = questionary.checkbox(
        "Select relevant threat categories for this project:",
        choices=[
            "data_poisoning",
            "backdoor_attacks",
            "adversarial_examples",
            "model_stealing",
            "membership_inference",
            "model_inversion"
        ]
    ).ask()

    # Create profile dictionary
    profile = {
        "threat_model": {
            "model_access": model_access,
            "attack_goal": attack_goal,
            "deployment_scenario": deployment_scenario,
            "data_sensitivity": data_sensitivity,
            "training_data_source": training_data_source,
            "model_type": model_type,
            "interface_exposed": interface_exposed,
            "threat_categories": threat_categories
        }
    }

    # Print as YAML
    print("\n=== Generated Threat Profile ===")
    print(yaml.dump(profile, sort_keys=False))

    # Optional: Ask to save
    save = questionary.confirm("Do you want to save this profile to a YAML file?").ask()
    if save:
        filename = questionary.text("Enter filename (e.g., profile.yaml):").ask()
        with open(f"../profiles/{filename}", "w") as f:
            yaml.dump(profile, f)
        print(f"✅ Profile saved as ../profiles/{filename}")

if __name__ == "__main__":
    run_questionnaire()
