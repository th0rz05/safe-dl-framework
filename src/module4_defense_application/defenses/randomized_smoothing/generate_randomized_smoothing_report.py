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
    lines.append("The histogram below shows the distribution of the Gaussian noise applied during training.\n")

    attack_type = data.get("attack", "unknown")
    hist_rel_path = f"noise_histograms/noise_distribution.png"
    hist_abs_path = os.path.join("results", "evasion", attack_type, hist_rel_path)
    if os.path.exists(hist_abs_path):
        lines.append(f"![Noise Distribution]({hist_rel_path})\n")

    lines.append("## Visual Examples")
    lines.append("Each example below compares the original and noisy version of a training sample.\n")

    examples_rel_path = f"noisy_examples"
    examples_abs_path = os.path.join("results", "evasion", attack_type, examples_rel_path)
    if os.path.exists(examples_abs_path):
        for fname in sorted(os.listdir(examples_abs_path)):
            if fname.endswith(".png"):
                title = fname.replace(".png", "").replace("_", " ").capitalize()
                lines.append(f"### {title}")
                lines.append(f"![{title}]({examples_rel_path}/{fname})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
