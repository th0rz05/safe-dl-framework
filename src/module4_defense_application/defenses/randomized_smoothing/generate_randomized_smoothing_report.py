import os
import json

def generate_randomized_smoothing_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Randomized Smoothing Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Randomized Smoothing")
    lines.append(f"**Noise Level (σ):** {data.get('sigma', 'N/A')}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n## Noise Distribution")
    lines.append("The following histogram shows the distribution of Gaussian noise added to the training samples.\n")

    attack_type = data.get("attack", "unknown")
    hist_path = f"results/evasion/{attack_type}/randomized_smoothing/histograms/noise_distribution.png"
    if os.path.exists(hist_path):
        lines.append(f"![Noise Distribution](histograms/noise_distribution.png)\n")

    lines.append("## Visual Examples\n")
    lines.append("The following examples show original vs. noisy training samples used in adversarial training.\n")

    examples_dir = f"results/evasion/{attack_type}/randomized_smoothing/noisy_examples"
    if os.path.exists(examples_dir):
        for fname in sorted(os.listdir(examples_dir)):
            if fname.endswith(".png"):
                title = fname.replace(".png", "").replace("_", " ").capitalize()
                lines.append(f"### {title}")
                lines.append(f"![{title}](noisy_examples/{fname})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
