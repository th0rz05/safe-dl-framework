import os
import json
import yaml
from datetime import datetime
from collections import defaultdict
from tabulate import tabulate
from pathlib import Path

def bucketize(value):
    if value < 0.33:
        return "Low"
    elif value < 0.66:
        return "Medium"
    else:
        return "High"

def get_report_path(attack_name, attack_type):
    base = "../../module2_attack_simulation/results"
    if attack_type == "data_poisoning":
        return f"{base}/data_poisoning/{attack_name}/{attack_name}_report.md"
    elif attack_type == "backdoor":
        return f"{base}/backdoor/{attack_name}/{attack_name}_report.md"
    elif attack_type == "evasion":
        return f"{base}/evasion/{attack_name}/{attack_name}_report.md"
    else:
        return "-"

def generate_recommendations(risk_data):
    recs = []

    for name, attack in risk_data.items():
        typ = attack["type"]
        score = attack["risk_score"]
        sev = attack["severity"]
        vis = attack["visibility"]

        # Evasion attacks
        if typ == "evasion":
            if score >= 1.5:
                recs.append(f"- **{name}**: Very high-risk evasion attack. Recommend adversarial training and randomized smoothing")
            elif sev >= 0.8 and vis <= 0.3:
                recs.append(f"- **{name}**: Stealthy but strong evasion attack. Suggest gradient masking and input preprocessing (e.g., JPEG compression).")
            elif vis >= 0.6:
                recs.append(f"- **{name}**: High-visibility evasion. Consider perturbation detection techniques.")

        # Data poisoning attacks
        elif typ == "data_poisoning":
            flip_rate = attack.get("flip_rate", 0.0)
            poison_frac = attack.get("fraction_poison", 0.0)
            if flip_rate > 0.05:
                recs.append(f"- **{name}**: Flip rate above 5%. Recommend data cleaning and per-class accuracy monitoring.")
            elif poison_frac > 0.1:
                recs.append(f"- **{name}**: Large poisoned subset detected. Use data provenance tracking or influence functions.")
            elif score > 0.5:
                recs.append(f"- **{name}**: Significant poisoning risk. Recommend robust loss functions or differentially private training.")

        # Backdoor attacks
        elif typ == "backdoor":
            blend = attack.get("blend_alpha", 1.0)
            if blend >= 1.0:
                recs.append(f"- **{name}**: Fully blended backdoor trigger. Use activation clustering and spectral signature defenses.")
            elif blend <= 0.3 and vis <= 0.3:
                recs.append(f"- **{name}**: Stealthy backdoor. Consider pruning and input anomaly detection.")
            if attack.get("asr", 0.0) > 0.8 and score >= 1.5:
                recs.append(f"- **{name}**: Backdoor with high ASR and high risk. Suggest fine-pruning or model inspection techniques.")

        # General fallback
        if score < 0.3:
            recs.append(f"- **{name}**: Low overall risk. No immediate action required, but monitor for future drift or attack evolution.")

    # Deduplicate if needed
    return list(dict.fromkeys(recs))

def generate_recommendation_tags(risk_data):
    recs = defaultdict(list)

    for name, attack in risk_data.items():
        typ = attack["type"]
        score = attack["risk_score"]
        sev = attack["severity"]
        vis = attack["visibility"]

        if typ == "evasion":
            if score >= 1.5:
                recs[name] += ["adversarial_training", "randomized_smoothing"]
            elif sev >= 0.8 and vis <= 0.3:
                recs[name] += ["gradient_masking", "jpeg_preprocessing"]
            elif vis >= 0.6:
                recs[name] += ["perturbation_detection"]

        elif typ == "data_poisoning":
            flip_rate = attack.get("flip_rate", 0.0)
            poison_frac = attack.get("fraction_poison", 0.0)
            if flip_rate > 0.05:
                recs[name] += ["data_cleaning", "per_class_monitoring"]
            elif poison_frac > 0.1:
                recs[name] += ["provenance_tracking", "influence_functions"]
            elif score > 0.5:
                recs[name] += ["robust_loss", "dp_training"]

        elif typ == "backdoor":
            blend = attack.get("blend_alpha", 1.0)
            if blend >= 1.0:
                recs[name] += ["activation_clustering", "spectral_signatures"]
            elif blend <= 0.3 and vis <= 0.3:
                recs[name] += ["anomaly_detection", "pruning"]
            if attack.get("asr", 0.0) > 0.8 and score >= 1.5:
                recs[name] += ["fine_pruning", "model_inspection"]

    return dict(recs)


def update_profile_with_recommendations(profile_path, recommendations):
    with open(profile_path, "r") as f:
        profile = yaml.safe_load(f)

    if "risk_analysis" not in profile:
        profile["risk_analysis"] = {}

    profile["risk_analysis"]["recommendations"] = recommendations

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile, f)

def update_profile_with_summary(profile_path, risk_data):
    with open(profile_path, "r") as f:
        profile = yaml.safe_load(f)

    if "risk_analysis" not in profile:
        profile["risk_analysis"] = {}

    summary = {}
    for name, info in risk_data.items():
        summary[name] = {
            "severity": round(info["severity"], 3),
            "probability": round(info["probability"], 3),
            "visibility": round(info["visibility"], 3),
            "risk_score": round(info["risk_score"], 3)
        }

    profile["risk_analysis"]["summary"] = summary

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile, f)


def generate_risk_report(risk_path: Path, profile_path: Path, report_path: Path):
    # Load files
    if not risk_path.exists():
        print(f"[!] File not found: {risk_path}")
        exit(1)

    if not profile_path.exists():
        print(f"[!] File not found: {profile_path}")
        exit(1)

    with open(risk_path, "r") as f:
        risk_data = json.load(f)

    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)

    # Build risk matrix and ranking
    matrix = defaultdict(lambda: defaultdict(list))
    ranking = []
    rows = []

    for attack, info in risk_data.items():
        sev = bucketize(info["severity"])
        prob = bucketize(info["probability"])
        matrix[sev][prob].append(attack)
        ranking.append((attack, info["risk_score"]))
        attack_type = info.get("type", "")
        report_link = get_report_path(attack,attack_type)
        rows.append([
            attack,
            attack_type,
            round(info["severity"], 2),
            round(info["probability"], 2),
            round(info["visibility"], 2),
            round(info["risk_score"], 2),
            f"[Report]({report_link})"
        ])

    ranking.sort(key=lambda x: x[1], reverse=True)

    # Generate markdown report
    lines = []

    lines.append("# Risk Analysis Report\n")
    lines.append(f"**Profile**: `{profile_path.name}`  ")
    lines.append(f"**Dataset**: `{profile_data['dataset']['name']}`  ")
    lines.append(f"**Model**: `{profile_data['model']['name']}`  ")
    lines.append(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Overview\n")
    lines.append("This report summarizes the risk associated with each attack simulated in Module 2 of the Safe-DL framework. "
                 "Each attack is evaluated based on its impact (severity), likelihood of success (probability), and perceptibility (visibility). "
                 "A final risk score is computed to help prioritize mitigation strategies.\n")

    lines.append("## Summary Table\n")
    lines.append(tabulate(rows, headers=["Attack", "Type", "Severity", "Probability", "Visibility", "Risk Score", "Report"], tablefmt="github"))
    lines.append("")

    lines.append("## Risk Matrix (Qualitative)\n")
    prob_levels = ["Low", "Medium", "High"]
    sev_levels = ["Low", "Medium", "High"]
    matrix_table = [["Severity \\ Probability"] + prob_levels]
    for sev in sev_levels:
        row = [sev]
        for prob in prob_levels:
            cell = ", ".join(matrix[sev][prob]) if matrix[sev][prob] else "-"
            row.append(cell)
        matrix_table.append(row)
    lines.append(tabulate(matrix_table, headers="firstrow", tablefmt="github"))
    lines.append("")

    lines.append("## Risk Ranking\n")
    for i, (attack, score) in enumerate(ranking, 1):
        atype = risk_data[attack]["type"]
        report_link = get_report_path(attack, atype)
        lines.append(f"{i}. **{attack}** — risk score: {score:.2f} → [Report]({report_link})")
    lines.append("")

    lines.append("## Recommendations\n")
    for rec in generate_recommendations(risk_data):
        lines.append(rec)

    # Save the report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[✓] Risk report saved to: {report_path}")

    # Group recommendations per attack
    raw_recs = generate_recommendations(risk_data)
    grouped = defaultdict(list)
    for line in raw_recs:
        if line.startswith("- **"):
            attack = line.split("**")[1]
            text = line.split("**: ")[1]
            grouped[attack].append(text)

    # Save into profile.yaml
    recommendation_tags = generate_recommendation_tags(risk_data)
    update_profile_with_recommendations(profile_path, recommendation_tags)
    print(f"[✓] Recommendations saved to: {profile_path} under `risk_analysis.recommendations`")

    update_profile_with_summary(profile_path, risk_data)
    print(f"[✓] Summary saved to: {profile_path} under `risk_analysis.summary`")

if __name__ == "__main__":
    risk_path = Path("results/risk_analysis.json")
    profile_path = Path("../profiles/test.yaml")
    report_path = Path("results/risk_report.md")

    generate_risk_report(risk_path, profile_path, report_path)