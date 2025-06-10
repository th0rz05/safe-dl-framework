# File: run_module6.py

import os
import sys
import questionary
from datetime import datetime
from glob import glob

# Import utility functions and constants from generate_report_utils.py
# Assuming generate_report_utils.py is in the same directory as run_module6.py
from generate_report_utils import (
    load_yaml,
    load_json, # Keeping this here as it's a general utility, though not directly used in run_module6.py's main logic.
    generate_report_header,
    PROFILES_DIR, # Imported constants
    REPORTS_DIR,  # CORRECTED: This will now point to ../reports
    # MODULE2_RESULTS_DIR, MODULE3_RESULTS_DIR etc. will be used by report generation functions later
)


# --- Utility Functions (Only those specific to run_module6.py's execution flow) ---
def select_profile_path() -> str:
    """
    Allows the user to select a threat profile.
    Returns the full path to the selected profile YAML file.
    """
    # PROFILES_DIR is imported from generate_report_utils.py
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
    # REPORTS_DIR is imported from generate_report_utils.py
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Generate output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_md_path = os.path.join(REPORTS_DIR, f"final_report_{timestamp}.md")

    print(f"\n[INFO] Generating final report for profile: {profile_name}")

    # --- NEXT STEP: Generate the first section of the report ---
    # Call the imported function to generate the report header
    report_lines.append(generate_report_header(profile_data, profile_name))

    # --- END OF REPORT ---
    final_report_content = "\n".join(report_lines)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_report_content)

    print(f"\n[✓] Final report generated successfully at: {output_md_path}")

if __name__ == "__main__":
    # Ensure necessary packages are installed
    try:
        import questionary
        import yaml # yaml is imported via generate_report_utils, but direct import here for robustness
    except ImportError:
        print("[!] Required packages 'questionary' and 'pyyaml' not found.")
        print("    Please install them using: pip install questionary pyyaml")
        sys.exit(1)

    main()