# File: run_module6.py

import os
import sys
import questionary
from datetime import datetime
from glob import glob

# Import utility functions and constants from generate_report_utils.py
from generate_report_utils import (
    load_yaml,
    load_json,
    generate_report_header,
    generate_system_details_section,
    generate_threat_profile_section,
    generate_attack_simulation_section,
    generate_risk_analysis_section,
    PROFILES_DIR,
    REPORTS_DIR,
    # MODULE2_RESULTS_DIR, MODULE3_RESULTS_DIR etc. will be used by report generation functions later
)


# --- Utility Functions (Only those specific to run_module6.py's execution flow) ---
def select_profile_path() -> str:
    """
    Allows the user to select a threat profile.
    Returns the full path to the selected profile YAML file.
    """
    profiles = glob(os.path.join(PROFILES_DIR, "*.yaml"))
    if not profiles:
        print(f"No profiles found in {PROFILES_DIR}/")
        return None

    profile_names = [os.path.basename(p) for p in profiles]
    selected_name = questionary.select(
        "Select a threat profile to use for the final report:",
        choices=profile_names
    ).ask()

    if selected_name:
        return os.path.join(PROFILES_DIR, selected_name)
    return None

def main():
    report_lines = []

    # 1. Profile Selection
    profile_path = select_profile_path()
    if not profile_path:
        print("No profile selected. Exiting.")
        return

    profile_data = load_yaml(profile_path)
    profile_name = os.path.basename(profile_path)

    # Ensure the reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Generate output file name without timestamp
    output_md_path = os.path.join(REPORTS_DIR, "final_report.md")

    print(f"\n[INFO] Generating final report for profile: {profile_name}")

    # Generate the initial sections of the report
    report_lines.append(generate_report_header(profile_data, profile_name))
    report_lines.append(generate_system_details_section(profile_data))
    report_lines.append(generate_threat_profile_section(profile_data))
    report_lines.append(generate_attack_simulation_section(profile_data))
    report_lines.append(generate_risk_analysis_section(profile_data))


    # --- END OF REPORT (for now) ---
    final_report_content = "\n".join(report_lines)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_report_content)

    print(f"\n[✓] Final report generated successfully at: {output_md_path}")

if __name__ == "__main__":
    try:
        import questionary
        import yaml
    except ImportError:
        print("[!] Required packages 'questionary' and 'pyyaml' not found.")
        print("    Please install them using: pip install questionary pyyaml")
        sys.exit(1)

    main()