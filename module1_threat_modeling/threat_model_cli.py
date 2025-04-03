import questionary
import yaml

def ask_with_help(question, choices, help_text):
    while True:
        choice = questionary.select(
            question,
            choices=choices + ["Help / What does this mean?"]
        ).ask()

        if choice == "Help / What does this mean?":
            print(help_text)
        else:
            return choice

def ask_checkbox_with_help(question, choices_with_explanations, preselected=None):
    while True:
        selected = questionary.checkbox(
            question,
            choices=[
                questionary.Choice(f"{k} – {v}", checked=(preselected and k in preselected))
                for k, v in choices_with_explanations.items()
            ] + [questionary.Choice("Help / What do these mean?")]
        ).ask()

        if any("Help" in s for s in selected):
            print("\nThreat category explanations:")
            for key, desc in choices_with_explanations.items():
                print(f"- {key}: {desc}")
        else:
            return [k for k in choices_with_explanations.keys() if any(k in s for s in selected)]

def run_questionnaire():
    print("=== Threat Modeling Questionnaire ===\n")

    model_access = ask_with_help(
        "What level of access might an attacker have to the model?",
        ["white-box", "gray-box", "black-box"],
        """
- white-box: Full access to architecture, weights, training data.
- gray-box: Partial knowledge (e.g. architecture but not weights).
- black-box: Access only to model inputs/outputs.
"""
    )

    attack_goal = ask_with_help(
        "What is the likely goal of the attacker?",
        ["targeted", "untargeted"],
        """
- targeted: The attacker wants a specific misclassification (e.g., always predict 'cat').
- untargeted: Any incorrect prediction is acceptable to the attacker.
"""
    )

    deployment_scenario = ask_with_help(
        "Where will the model be deployed?",
        ["cloud", "edge", "mobile", "api_public", "on_device"],
        """
- cloud: Running in a server or datacenter.
- edge: On local devices near users (e.g., IoT cameras).
- mobile: In mobile apps.
- api_public: Exposed to the internet via an API.
- on_device: Embedded in firmware or local software.
"""
    )

    data_sensitivity = ask_with_help(
        "How sensitive is the training data?",
        ["high", "medium", "low"],
        """
- high: Data includes personal, biometric, or confidential information.
- medium: Some relevance, but not critical.
- low: Public or synthetic datasets.
"""
    )

    training_data_source = ask_with_help(
        "Where does the training data come from?",
        ["internal_clean", "external_public", "user_generated", "mixed"],
        """
- internal_clean: Fully controlled and verified by your team.
- external_public: Open datasets from the internet.
- user_generated: Uploaded or labeled by external users.
- mixed: A combination of the above.
"""
    )

    model_type = ask_with_help(
        "What type of architecture will the model use?",
        ["cnn", "transformer", "mlp", "other"],
        """
- cnn: Convolutional Neural Network (image classification, etc).
- transformer: Vision transformers, BERT-like models.
- mlp: Simple feed-forward neural networks.
- other: RNNs, GNNs, etc.
"""
    )

    interface_exposed = ask_with_help(
        "How will the model be accessed by users or systems?",
        ["api", "local_app", "sdk", "none"],
        """
- api: Model is served via an online API.
- local_app: Used inside a desktop or mobile app.
- sdk: Distributed as a library/module.
- none: Only embedded, not exposed directly.
"""
    )

    # === Threat suggestion logic ===
    suggested_threats = set()

    if training_data_source in ["external_public", "user_generated", "mixed"]:
        suggested_threats.add("data_poisoning")

    if interface_exposed == "api":
        suggested_threats.update(["model_stealing", "membership_inference"])

    if deployment_scenario in ["mobile", "edge", "api_public"]:
        suggested_threats.add("adversarial_examples")

    if model_access in ["white-box", "gray-box"]:
        suggested_threats.add("backdoor_attacks")

    if data_sensitivity == "high":
        suggested_threats.add("model_inversion")

    print("\nBased on your answers, we suggest the following threat categories:")
    for t in suggested_threats:
        print(f"- {t}")

    edit = questionary.confirm("Do you want to edit this list?", default=False).ask()

    threat_category_explanations = {
        "data_poisoning": "Malicious data inserted in training to corrupt the model.",
        "backdoor_attacks": "Triggers planted during training to cause misclassification when activated.",
        "adversarial_examples": "Inputs crafted to cause errors at inference time.",
        "model_stealing": "Cloning the model by observing its outputs.",
        "membership_inference": "Determining if a specific input was in the training data.",
        "model_inversion": "Reconstructing private inputs using model predictions."
    }

    if edit:
        threat_categories = ask_checkbox_with_help(
            "Select relevant threat categories (space to toggle):",
            threat_category_explanations,
            preselected=suggested_threats
        )
    else:
        threat_categories = list(suggested_threats)

    # Create final profile
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

    # Show result
    print("\n=== Generated Threat Profile ===")
    print(yaml.dump(profile, sort_keys=False))

    # Save
    save = questionary.confirm("Do you want to save this profile to a YAML file?").ask()
    if save:
        filename = questionary.text("Enter filename (e.g., 'test' to save as test.yaml):").ask()
        with open(f"../profiles/{filename}.yaml", "w") as f:
            yaml.dump(profile, f)
        print(f"\n Profile saved as ../profiles/{filename}.yaml")

if __name__ == "__main__":
    run_questionnaire()
