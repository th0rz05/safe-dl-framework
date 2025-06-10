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