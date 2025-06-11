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

def load_risk_analysis_results() -> dict:
    """
    Loads the risk_analysis.json file from the MODULE3_RESULTS_DIR.
    Returns:
        dict: The loaded risk analysis data.
    Raises:
        FileNotFoundError: If the risk_analysis.json file is not found.
    """
    risk_analysis_path = os.path.join(MODULE3_RESULTS_DIR, "risk_analysis.json")
    if not os.path.exists(risk_analysis_path):
        raise FileNotFoundError(f"Risk analysis results not found: {risk_analysis_path}. "
                                f"Please ensure Module 3 has been run and 'risk_analysis.json' exists.")
    return load_json(risk_analysis_path)

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
            section_lines.append(
                "  \n*Note: While listed in the threat profile, 'model_stealing', 'membership_inference', "
                "and 'model_inversion' attack simulations and their corresponding defenses are "
                "currently considered future work and are not yet fully implemented in subsequent Modules.*")
        else:
            section_lines.append(f"- **{display_name}**: `{value}`")
            section_lines.append(f"  *{description}*")
        section_lines.append("") # Add an empty line for spacing between items

    return "\n".join(section_lines)


def generate_attack_simulation_section(profile_data: dict) -> str:
    """
    Generates a Markdown section summarizing adversarial attack simulations (Module 2).
    Focuses on Data Poisoning (Clean Label, Label Flipping) and Backdoor attacks.
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
                     "Attack Metric", "Key Parameters", "Full Results"]  # 'Attack Metric' added
    table_lines = [
        "| " + " | ".join(table_headers) + " |",
        "|:----------------|:--------------|:------------------------|:---------------------|:--------------|:---------------|:-------------|"
    ]

    # Process Data Poisoning attacks
    data_poisoning_attacks = attack_overrides.get('data_poisoning', {})

    for method, params in data_poisoning_attacks.items():
        metrics_file_path = os.path.join(MODULE2_RESULTS_DIR, 'data_poisoning', method, f"{method}_metrics.json")

        report_md_file_name = f"{method}_report.md"
        report_file_path_md = os.path.join(MODULE2_RESULTS_DIR, 'data_poisoning', method, report_md_file_name)

        current_dir = os.getcwd()
        reports_abs_path = os.path.join(current_dir, REPORTS_DIR)

        # Default link to JSON, will be updated to MD if MD exists
        link_to_details_path = os.path.relpath(metrics_file_path, reports_abs_path)

        if not os.path.exists(metrics_file_path):
            print(f"[!] Warning: Metrics file not found for data_poisoning/{method}: {metrics_file_path}")
            continue

        if os.path.exists(report_file_path_md):
            link_to_details_path = os.path.relpath(report_file_path_md, reports_abs_path)
        else:
            print(
                f"[!] Warning: Markdown report file not found for data_poisoning/{method}: {report_file_path_md}. Linking to JSON instead.")

        attack_metrics = load_json(metrics_file_path)

        attack_category = "Data Poisoning"
        attack_method_display = method.replace('_', ' ').title()

        clean_acc_after_attack = attack_metrics.get("accuracy_after_attack", None)  # Changed var name for clarity

        clean_acc_pre_attack_display = f"{baseline_accuracy * 100:.2f}%"

        if clean_acc_after_attack is not None:
            impact_on_clean_acc_display = f"{clean_acc_after_attack * 100:.2f}%"
            # For data poisoning, 'Attack Metric' is the accuracy after attack (i.e., degraded accuracy)
            attack_metric_display = f"{clean_acc_after_attack * 100:.2f}% (Degraded Acc.)"
        else:
            impact_on_clean_acc_display = "N/A"
            attack_metric_display = "N/A"

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
            attack_metric_display,  # Now includes Post-Poison Acc. for consistency
            ", ".join(key_params_str) if key_params_str else "N/A",
            f"[Details]({link_to_details_path})"
        ]
        table_lines.append("| " + " | ".join(table_row) + " |")

    # Process Backdoor attacks
    backdoor_attacks = attack_overrides.get('backdoor', {})

    for method, params in backdoor_attacks.items():
        metrics_file_path = os.path.join(MODULE2_RESULTS_DIR, 'backdoor', method, f"{method}_metrics.json")

        report_md_file_name = f"{method}_report.md"
        report_file_path_md = os.path.join(MODULE2_RESULTS_DIR, 'backdoor', method, report_md_file_name)

        current_dir = os.getcwd()
        reports_abs_path = os.path.join(current_dir, REPORTS_DIR)

        # Default link to JSON, will be updated to MD if MD exists
        link_to_details_path = os.path.relpath(metrics_file_path, reports_abs_path)

        if not os.path.exists(metrics_file_path):
            print(f"[!] Warning: Metrics file not found for backdoor/{method}: {metrics_file_path}")
            continue

        if os.path.exists(report_file_path_md):
            link_to_details_path = os.path.relpath(report_file_path_md, reports_abs_path)
        else:
            print(
                f"[!] Warning: Markdown report file not found for backdoor/{method}: {report_file_path_md}. Linking to JSON instead.")

        attack_metrics = load_json(metrics_file_path)

        attack_category = "Backdoor"
        attack_method_display = method.replace('_', ' ').title()

        # For backdoor, these are from the backdoor JSON
        clean_acc_post_backdoor = attack_metrics.get("accuracy_clean_testset", None)
        attack_success_rate = attack_metrics.get("attack_success_rate", None)

        clean_acc_pre_attack_display = f"{baseline_accuracy * 100:.2f}%"

        if clean_acc_post_backdoor is not None:
            impact_on_clean_acc_display = f"{clean_acc_post_backdoor * 100:.2f}%"
        else:
            impact_on_clean_acc_display = "N/A"

        if attack_success_rate is not None:
            attack_metric_display = f"{attack_success_rate * 100:.2f}% (ASR)"
        else:
            attack_metric_display = "N/A (ASR)"

        key_params_str = []
        if method == 'static_patch':
            key_params_str.append(f"Poison Frac.: {params.get('poison_fraction', 'N/A')}")
            key_params_str.append(f"Target Class: {params.get('target_class', 'N/A')}")
            key_params_str.append(f"Patch Type: {params.get('patch_type', 'N/A')}")
        elif method == 'learned_trigger':
            key_params_str.append(f"Poison Frac.: {params.get('poison_fraction', 'N/A')}")
            key_params_str.append(f"Target Class: {params.get('target_class', 'N/A')}")
            key_params_str.append(f"Epochs Trigger: {params.get('epochs_trigger', 'N/A')}")

        table_row = [
            attack_category,
            attack_method_display,
            clean_acc_pre_attack_display,
            impact_on_clean_acc_display,
            attack_metric_display,
            ", ".join(key_params_str) if key_params_str else "N/A",
            f"[Details]({link_to_details_path})"
        ]
        table_lines.append("| " + " | ".join(table_row) + " |")

    # Process Evasion attacks
    evasion_attacks = attack_overrides.get('evasion', {})

    for method, params in evasion_attacks.items():
        metrics_file_path = os.path.join(MODULE2_RESULTS_DIR, 'evasion', method, f"{method}_metrics.json")

        report_md_file_name = f"{method}_report.md"
        report_file_path_md = os.path.join(MODULE2_RESULTS_DIR, 'evasion', method, report_md_file_name)

        current_dir = os.getcwd()
        reports_abs_path = os.path.join(current_dir, REPORTS_DIR)

        # Default link to JSON, will be updated to MD if MD exists
        link_to_details_path = os.path.relpath(metrics_file_path, reports_abs_path)

        if not os.path.exists(metrics_file_path):
            print(f"[!] Warning: Metrics file not found for evasion/{method}: {metrics_file_path}")
            continue

        if os.path.exists(report_file_path_md):
            link_to_details_path = os.path.relpath(report_file_path_md, reports_abs_path)
        else:
            print(
                f"[!] Warning: Markdown report file not found for evasion/{method}: {report_file_path_md}. Linking to JSON instead.")

        attack_metrics = load_json(metrics_file_path)

        attack_category = "Evasion"
        attack_method_display = method.replace('_', ' ').title()

        clean_acc_post_evasion = attack_metrics.get("accuracy_clean_testset", None)
        adversarial_accuracy = attack_metrics.get("accuracy_adversarial_testset", None)

        clean_acc_pre_attack_display = f"{baseline_accuracy * 100:.2f}%"

        if clean_acc_post_evasion is not None:
            impact_on_clean_acc_display = f"{clean_acc_post_evasion * 100:.2f}%"
        else:
            impact_on_clean_acc_display = "N/A"

        if adversarial_accuracy is not None:
            attack_metric_display = f"{adversarial_accuracy * 100:.2f}% (Adv. Acc.)"  # Lower is better for attacker
        else:
            attack_metric_display = "N/A (Adv. Acc.)"

        key_params_str = []
        if method == 'pgd':
            key_params_str.append(f"Epsilon: {params.get('epsilon', 'N/A')}")
            key_params_str.append(f"Num Iter: {params.get('num_iter', 'N/A')}")
        elif method == 'spsa':
            key_params_str.append(f"Epsilon: {params.get('epsilon', 'N/A')}")
            key_params_str.append(f"Num Steps: {params.get('num_steps', 'N/A')}")
            key_params_str.append(f"Delta: {params.get('delta', 'N/A')}")  # Added delta
        elif method == 'fgsm':
            key_params_str.append(f"Epsilon: {params.get('epsilon', 'N/A')}")
        elif method == 'cw':
            key_params_str.append(f"Confidence: {params.get('confidence', 'N/A')}")
            key_params_str.append(f"Max Iter: {params.get('max_iterations', 'N/A')}")
        elif method == 'deepfool':
            key_params_str.append(f"Max Iter: {params.get('max_iter', 'N/A')}")
            key_params_str.append(f"Overshoot: {params.get('overshoot', 'N/A')}")
        elif method == 'nes':
            key_params_str.append(f"Epsilon: {params.get('epsilon', 'N/A')}")
            key_params_str.append(f"Num Queries: {params.get('num_queries', 'N/A')}")
        elif method == 'transfer':  # Assuming 'transfer_fgsm' might just be 'transfer' in profile
            key_params_str.append(f"Substitute Model: {params.get('substitute_model', {}).get('name', 'N/A')}")
            key_params_str.append(f"Epsilon: {params.get('epsilon', 'N/A')}")

        table_row = [
            attack_category,
            attack_method_display,
            clean_acc_pre_attack_display,
            impact_on_clean_acc_display,
            attack_metric_display,
            ", ".join(key_params_str) if key_params_str else "N/A",
            f"[Details]({link_to_details_path})"
        ]
        table_lines.append("| " + " | ".join(table_row) + " |")

    section_lines.extend(table_lines)

    section_lines.append(
        "\n**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. "
        "'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being subjected to the attack (e.g., trained with poisoned data, or backdoor injected). "
        "For Data Poisoning attacks, 'Attack Metric' displays the degraded accuracy of the model on clean inputs after poisoning. "
        "For Backdoor attacks, 'Attack Metric' displays the Attack Success Rate (ASR), indicating the percentage of adversarial samples (with trigger) successfully misclassified to the target class. "
        "For Evasion attacks, 'Attack Metric' displays the Adversarial Accuracy (Adv. Acc.) on perturbed inputs, where a lower value indicates a more successful attack.\n"
    )

    section_lines.append("\n")

    return "\n".join(section_lines)


def generate_risk_analysis_section(profile_data: dict) -> str:
    """
    Generates the Risk Analysis section for the final report, including summary table,
    risk matrix, ranking, and recommendations. It filters the risk_data to
    only include attacks specified in the profile's attack_overrides.
    """
    section_lines = [
        "## 5. Risk Analysis (Module 3)\n",
        "This section summarizes the risk assessment performed on the simulated attacks. "
        "Each attack is evaluated based on its severity, probability, and visibility. "
        "A final risk score is computed to help prioritize mitigation strategies, "
        "followed by specific defense recommendations.\n"
    ]

    risk_data = load_risk_analysis_results()

    # --- Step 1: Filter risk_data based on attack_overrides in profile_data ---
    # NEW LOGIC: Handle the nested dictionary structure of attack_overrides
    profile_attack_overrides = profile_data.get('attack_overrides', {})

    # Extract names of attacks defined in attack_overrides from the nested structure
    filtered_attack_names = set()
    for attack_type, attacks_by_type in profile_attack_overrides.items():
        if isinstance(attacks_by_type, dict):  # Ensure it's a dict before iterating keys
            for attack_name in attacks_by_type.keys():
                filtered_attack_names.add(attack_name)

    # Create a new dictionary containing only the risk data for relevant attacks
    filtered_risk_data = {
        name: data for name, data in risk_data.items()
        if name in filtered_attack_names
    }

    # Use filtered_risk_data for all subsequent sections
    current_risk_data = filtered_risk_data

    # 5.1 Summary Table
    section_lines.append("\n### 5.1 Risk Summary Table\n")
    if not current_risk_data:
        section_lines.append(
            "No risk analysis data available for the attacks specified in the profile's `attack_overrides`.")
    else:
        headers = ["Attack", "Type", "Severity", "Probability", "Visibility", "Risk Score", "Report"]
        table_rows = []
        for attack_name, data in current_risk_data.items():
            attack_type_folder = data.get('type')
            if attack_type_folder == 'data_poisoning':
                attack_type_folder_path = 'data_poisoning'
            elif attack_type_folder == 'backdoor':
                attack_type_folder_path = 'backdoor'
            elif attack_type_folder == 'evasion':
                attack_type_folder_path = 'evasion'
            else:
                attack_type_folder_path = 'unknown'  # Fallback

            report_link = os.path.join(
                "..", "module2_attack_simulation", "results",
                attack_type_folder_path, attack_name, f"{attack_name}_report.md"
            ).replace("\\", "/")

            table_rows.append([
                attack_name.replace('_', ' ').title(),
                data.get('type', 'N/A').replace('_', ' ').title(),
                f"{data.get('severity', 0.0):.2f}",
                f"{data.get('probability', 0.0):.2f}",
                f"{data.get('visibility', 0.0):.2f}",
                f"{data.get('risk_score', 0.0):.2f}",
                f"[Details]({report_link})"
            ])

        table_rows.sort(key=lambda x: float(x[5]), reverse=True)

        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "|:--------------" * len(headers) + "|\n"
        for row in table_rows:
            table_str += "| " + " | ".join(row) + " |\n"
        section_lines.append(table_str)

    # 5.2 Risk Matrix (Qualitative)
    section_lines.append("\n### 5.2 Risk Matrix (Qualitative)\n")
    section_lines.append(
        "This matrix categorizes attacks based on their qualitative Severity and Probability levels.\n")

    def bucketize_qualitative(value):
        if value < 0.4:
            return "Low"
        elif value < 0.7:
            return "Medium"
        else:
            return "High"

    matrix = {
        "High": {"High": [], "Medium": [], "Low": []},
        "Medium": {"High": [], "Medium": [], "Low": []},
        "Low": {"High": [], "Medium": [], "Low": []},
    }

    if current_risk_data:
        for attack_name, data in current_risk_data.items():
            severity_bucket = bucketize_qualitative(data.get('severity', 0.0))
            probability_bucket = bucketize_qualitative(data.get('probability', 0.0))
            if severity_bucket in matrix and probability_bucket in matrix[severity_bucket]:
                matrix[severity_bucket][probability_bucket].append(attack_name.replace('_', ' ').title())

    matrix_headers = ["Severity \\ Probability", "Low", "Medium", "High"]
    matrix_table_str = "| " + " | ".join(matrix_headers) + " |\n"
    matrix_table_str += "|:-----------------------" * len(matrix_headers) + "|\n"

    severity_order = ["High", "Medium", "Low"]
    probability_order = ["Low", "Medium", "High"]

    for sev_bucket in severity_order:
        row_content = [sev_bucket]
        for prob_bucket in probability_order:
            attacks_in_bucket = ", ".join(sorted(matrix[sev_bucket][prob_bucket]))
            row_content.append(attacks_in_bucket if attacks_in_bucket else "-")
        matrix_table_str += "| " + " | ".join(row_content) + " |\n"
    section_lines.append(matrix_table_str)

    # 5.3 Risk Ranking
    section_lines.append("\n### 5.3 Risk Ranking\n")
    section_lines.append("Attacks ranked by their calculated Risk Score, from highest to lowest.\n")

    if current_risk_data:
        ranked_attacks = []
        for attack_name, data in current_risk_data.items():
            attack_type_folder = data.get('type')
            if attack_type_folder == 'data_poisoning':
                attack_type_folder_path = 'data_poisoning'
            elif attack_type_folder == 'backdoor':
                attack_type_folder_path = 'backdoor'
            elif attack_type_folder == 'evasion':
                attack_type_folder_path = 'evasion'
            else:
                attack_type_folder_path = 'unknown'

            report_link = os.path.join(
                "..", "module2_attack_simulation", "results",
                attack_type_folder_path, attack_name, f"{attack_name}_report.md"
            ).replace("\\", "/")

            ranked_attacks.append((
                attack_name.replace('_', ' ').title(),
                data.get('risk_score', 0.0),
                report_link
            ))

        ranked_attacks.sort(key=lambda x: x[1], reverse=True)

        for i, (name, score, link) in enumerate(ranked_attacks):
            section_lines.append(f"{i + 1}. **{name}** — Risk Score: {score:.2f} → [Details]({link})")
    else:
        section_lines.append("No attacks to rank based on the profile's `attack_overrides`.")
    section_lines.append("\n")

    # 5.4 Defense Recommendations
    section_lines.append("\n### 5.4 Defense Recommendations\n")
    section_lines.append(
        "Based on the identified risks and threat profile, the following defense recommendations are provided:\n")

    all_recommendations = profile_data.get('risk_analysis', {}).get('recommendations', {})

    # Filter recommendations to only include those for attacks in filtered_attack_names
    filtered_recommendations = {
        attack_name: rec_list
        for attack_name, rec_list in all_recommendations.items()
        if attack_name in filtered_attack_names
    }

    if filtered_recommendations:
        for attack_name, rec_list in filtered_recommendations.items():
            section_lines.append(f"- **{attack_name.replace('_', ' ').title()}**:")
            for rec in rec_list:
                if not rec.strip().startswith('-'):
                    section_lines.append(f"  - {rec}")
                else:
                    section_lines.append(f"  {rec}")
    else:
        section_lines.append("No specific defense recommendations found for the selected attacks in the profile data.")
    section_lines.append("\n")

    # 5.5 Paths to Details
    section_lines.append("\n### 5.5 Paths to Details\n")
    section_lines.append(
        "For more in-depth information about individual attacks, including raw metrics, "
        "attack visualizations, and specific parameters, please refer to the detailed "
        "reports linked in the 'Risk Summary Table' and 'Risk Ranking' sections above.\n"
    )

    return "\n".join(section_lines)
