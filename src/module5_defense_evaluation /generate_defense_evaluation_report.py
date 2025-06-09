import os
import json
import yaml
from datetime import datetime

def generate_markdown_table(data):
    lines = []
    lines.append("| Attack | Defense | Mitigation | CAD | Cost | Final Score |")
    lines.append("|--------|---------|------------|-----|------|--------------|")

    for attack_category in data:
        for attack_name in data[attack_category]:
            for defense_name, scores in data[attack_category][attack_name].items():
                mitigation = scores.get("mitigation_score", "N/A")
                cad = scores.get("cad_score", "N/A")
                cost = scores.get("defense_cost_score", "N/A")
                final = scores.get("final_score", "N/A")
                lines.append(f"| {attack_name} | {defense_name} | {mitigation:.3f} | {cad:.3f} | {cost:.3f} | {final:.3f} |")
    return "\n".join(lines)


def generate_report(profile_path,json_path,md_path) -> None:
    if not os.path.exists(json_path):
        print(f"[!] Input file not found: {json_path}")
        return

    if not os.path.exists(profile_path):
        print(f"[!] Profile file not found: {profile_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)


    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []


    report_lines.append("# Defense Evaluation Report\n")
    report_lines.append(f"**Profile**: `{profile_path.name}`  ")
    report_lines.append(f"**Dataset**: `{profile_data['dataset']['name']}`  ")
    report_lines.append(f"**Model**: `{profile_data['model']['name']}`  ")
    report_lines.append(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report_lines.append("## Overview\n")
    report_lines.append("This report summarizes the risk associated with each attack simulated in Module 2 of the Safe-DL framework. "
                 "Each attack is evaluated based on its impact (severity), likelihood of success (probability), and perceptibility (visibility). "
                 "A final risk score is computed to help prioritize mitigation strategies.\n")

    report_lines.append("## Summary Table\n")
    report_lines.append(generate_markdown_table(data))

    report_lines.append("\n## Notes\n")
    report_lines.append("- Mitigation Score: Improvement in model accuracy or class precision due to the defense.")
    report_lines.append("- CAD (Clean Accuracy Drop): Trade-off cost on clean data.")
    report_lines.append("- Cost Score: Relative computational/resource impact.")
    report_lines.append("- Final Score: Aggregated score considering all metrics.\n")

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"[✓] Report generated at: {md_path}")

if __name__ == "__main__":
    generate_report()
