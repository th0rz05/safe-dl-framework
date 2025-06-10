# File: generate_report_utils.py

import os
import json
import yaml
from datetime import datetime

# --- Configuration Paths (relative to module6_reporting/) ---
PROFILES_DIR = "../profiles"
MODULE2_RESULTS_DIR = "../module2_attack_simulation/results"
MODULE3_RESULTS_DIR = "../module3_risk_analysis/results"
MODULE4_RESULTS_DIR = "../module4_defense_application/results"
MODULE5_RESULTS_DIR = "../module5_defense_evaluation/results"
REPORTS_DIR = "../reports" # Corrected path for final reports


# --- Utility Functions for Data Loading ---
def load_yaml(path: str) -> dict:
    """
    Loads a YAML file from the given path.
    Args:
        path (str): The full path to the YAML file.
    Returns:
        dict: The loaded YAML data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path: str) -> dict:
    """
    Loads a JSON file from the given path.
    Args:
        path (str): The full path to the JSON file.
    Returns:
        dict: The loaded JSON data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_baseline_accuracy() -> float:
    """
    Loads the overall_accuracy from the baseline_accuracy.json file.
    Returns:
        float: The overall accuracy from the baseline, or 0.0 if not found.
    """
    baseline_path = os.path.join(MODULE2_RESULTS_DIR, "baseline_accuracy.json")
    if os.path.exists(baseline_path):
        data = load_json(baseline_path)
        return data.get("overall_accuracy", 0.0)
    print(f"[!] Warning: Baseline accuracy file not found at {baseline_path}. Returning 0.0.")
    return 0.0

# --- Report Section Generation Functions ---
def generate_report_header(profile_data: dict, profile_name: str) -> str:
    """
    Generates the initial header section of the final report.
    Args:
        profile_data (dict): The loaded data from the selected threat profile YAML.
        profile_name (str): The base name of the selected profile file (e.g., "my_profile.yaml").
    Returns:
        str: A Markdown string representing the report header.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_lines = [
        f"# Safe-DL Framework - Final Security Report",
        f"**Profile Selected**: `{profile_name}`",
        f"**Report Generated On**: {now}",
        "\n---", # Horizontal rule
        "\n## 1. Introduction and Overview",
        "This comprehensive report aggregates the findings from the Safe-DL framework's security assessment, "
        "covering threat modeling, attack simulation, risk analysis, defense application, and defense evaluation. "
        "Its purpose is to provide a unified overview of the deep learning model's security posture against "
        "adversarial threats, document the mitigation strategies applied, and quantify their effectiveness. "
        "This dossier serves as a critical resource for decision-making regarding model deployment and "
        "continuous security improvement.\n"
    ]
    return "\n".join(header_lines)

def generate_system_details_section(profile_data: dict) -> str:
    """
    Generates a Markdown section detailing the model and dataset from the profile.
    Args:
        profile_data (dict): The loaded data from the selected threat profile YAML.
    Returns:
        str: A Markdown string representing the system details section.
    """
    model_details = profile_data.get('model', {})
    dataset_details = profile_data.get('dataset', {})

    # Extract model details
    model_name = model_details.get('name', 'N/A')
    model_type = model_details.get('type', 'N/A')
    model_input_shape = model_details.get('input_shape', 'N/A')
    model_num_classes = model_details.get('num_classes', 'N/A')
    model_params = model_details.get('params', 'N/A') # Including params as it's in your example

    # Extract dataset details
    dataset_name = dataset_details.get('name', 'N/A')
    dataset_type = dataset_details.get('type', 'N/A')

    section_lines = [
        "## 2. System Under Evaluation Details",
        "This section details the core components of the system analyzed in this report, as defined in the selected threat profile.\n",
        "### 2.1 Model Details",
        f"- **Name**: `{model_name}`",
        f"- **Type**: `{model_type}`",
        f"- **Input Shape**: `{model_input_shape}`",
        f"- **Number of Classes**: `{model_num_classes}`"
    ]
    if model_params != 'N/A': # Only add if params exist
        section_lines.append(f"- **Parameters**: `{model_params}`")
    section_lines.append("\n") # Add a newline for spacing

    section_lines.extend([
        "### 2.2 Dataset Details",
        f"- **Name**: `{dataset_name}`",
        f"- **Type**: `{dataset_type}`\n"
    ])
    return "\n".join(section_lines)

def generate_threat_profile_section(profile_data: dict) -> str:
    """
    Generates a Markdown section detailing the threat model from the profile.
    Args:
        profile_data (dict): The loaded data from the selected threat profile YAML.
    Returns:
        str: A Markdown string representing the threat profile section.
    """
    threat_model = profile_data.get('threat_model', {})

    section_lines = [
        "## 3. Threat Profile (Module 1)",
        "This section outlines the specific characteristics of the system's environment and the anticipated adversary, "
        "as defined during the threat modeling phase (Module 1). These parameters guide the subsequent attack simulations, "
        "risk analysis, and defense applications.\n"
    ]

    # Define a mapping for display names and optional descriptions
    # Using ordered list of tuples to maintain a consistent order in the report
    threat_model_fields = [
        ("model_access", "Model Access", "Describes the level of access the adversary has to the model internals (e.g., weights, architecture)."),
        ("attack_goal", "Attack Goal", "Defines the adversary's objective (e.g., targeted misclassification, untargeted denial of service)."),
        ("deployment_scenario", "Deployment Scenario", "Indicates where the model is deployed (e.g., cloud, edge device, mobile)."),
        ("interface_exposed", "Interface Exposed", "How the model interacts with external entities (e.g., API, direct access, web application)."),
        ("model_type", "Model Type", "The architectural type of the deep learning model."),
        ("data_sensitivity", "Data Sensitivity", "The sensitivity level of the data used by the model, impacting privacy concerns."),
        ("training_data_source", "Training Data Source", "Origin and cleanliness of the data used for training the model."),
        ("threat_categories", "Threat Categories", "A list of attack types considered relevant for this threat profile.")
    ]

    for key, display_name, description in threat_model_fields:
        value = threat_model.get(key, 'N/A')
        if key == "threat_categories":
            section_lines.append(f"- **{display_name}**:")
            if isinstance(value, list) and value:
                for item in value:
                    section_lines.append(f"    - `{item}`")
            else:
                section_lines.append(f"    - `N/A`")
            section_lines.append("")
            section_lines.append(f"  *{description}*")
        else:
            section_lines.append(f"- **{display_name}**: `{value}`")
            section_lines.append(f"  *{description}*")
        section_lines.append("") # Add an empty line for spacing between items

    return "\n".join(section_lines)


def generate_attack_simulation_section(profile_data: dict) -> str:
    """
    Generates a Markdown section summarizing adversarial attack simulations (Module 2).
    Focuses on Data Poisoning (Clean Label, Label Flipping) initially.
    """
    section_lines = [
        "## 4. Attack Simulation (Module 2)",
        "This section summarizes the outcomes of the adversarial attack simulations performed against the model "
        "based on the defined threat profile. These simulations quantify the model's vulnerability to various "
        "attack types before any defenses are applied.\n",
        "### 4.1 Overview of Simulated Attacks\n"
    ]

    attack_overrides = profile_data.get('attack_overrides', {})

    baseline_accuracy = load_baseline_accuracy()
    if baseline_accuracy == 0.0:
        section_lines.append("[!] Warning: Baseline accuracy not found or is 0.0. Attack metrics may be misleading.\n")

    table_headers = ["Attack Category", "Attack Method", "Clean Acc. (Pre-Attack)", "Impact on Clean Acc.",
                     "Key Parameters", "Full Results"]
    table_lines = [
        "| " + " | ".join(table_headers) + " |",
        "|:----------------|:--------------|:------------------------|:---------------------|:---------------|:-------------|"
    ]

    data_poisoning_attacks = attack_overrides.get('data_poisoning', {})

    for method, params in data_poisoning_attacks.items():
        metrics_file_path = os.path.join(MODULE2_RESULTS_DIR, 'data_poisoning', method, f"{method}_metrics.json")

        current_dir = os.getcwd()
        reports_abs_path = os.path.join(current_dir, REPORTS_DIR)
        relative_results_path = os.path.relpath(metrics_file_path, reports_abs_path)

        if not os.path.exists(metrics_file_path):
            print(f"[!] Warning: Metrics file not found for data_poisoning/{method}: {metrics_file_path}")
            continue

        attack_metrics = load_json(metrics_file_path)

        attack_category = "Data Poisoning"
        attack_method_display = method.replace('_', ' ').title()

        clean_acc_after_attack_prep = attack_metrics.get("accuracy_after_attack", None)

        # "Clean Acc. (Pre-Attack)" é a precisão do modelo limpo ANTES de ser envenenado (i.e., a baseline)
        clean_acc_pre_attack_display = f"{baseline_accuracy * 100:.2f}%"

        # "Impact on Clean Acc." é a precisão do modelo APÓS o envenenamento
        if clean_acc_after_attack_prep is not None:
            impact_on_clean_acc_display = f"{clean_acc_after_attack_prep * 100:.2f}%"
        else:
            impact_on_clean_acc_display = "N/A"

        key_params_str = []
        if method == 'clean_label':
            key_params_str.append(f"Poison Fraction: {params.get('fraction_poison', 'N/A')}")
            key_params_str.append(f"Target Class: {params.get('target_class', 'N/A')}")
        elif method == 'label_flipping':
            key_params_str.append(f"Flip Rate: {params.get('flip_rate', 'N/A')}")
            key_params_str.append(f"Target Class: {params.get('target_class', 'N/A')}")

        table_row = [
            attack_category,
            attack_method_display,
            clean_acc_pre_attack_display,
            impact_on_clean_acc_display,
            ", ".join(key_params_str) if key_params_str else "N/A",
            f"[Details]({relative_results_path})"
        ]
        table_lines.append("| " + " | ".join(table_row) + " |")

    section_lines.extend(table_lines)

    section_lines.append(
        "\n**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. 'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being trained with poisoned data, reflecting the attack's effectiveness in degrading performance.\n")

    section_lines.append("\n")

    return "\n".join(section_lines)