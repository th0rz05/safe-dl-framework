import json
import os

def generate_static_patch_report(json_file, md_file):
    with open(json_file, "r") as f:
        results = json.load(f)

    lines = []

    lines.append("# Backdoor Attack Report — Static Patch\n")

    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {results.get('attack_type')}")
    lines.append(f"- **Patch Type:** {results.get('patch_type')}")
    lines.append(f"- **Patch Size Ratio:** {results.get('patch_size_ratio')}")
    lines.append(f"- **Patch Position:** {results.get('patch_position')}")
    lines.append(f"- **Poisoned Fraction:** {results.get('poison_fraction')}")
    lines.append(f"- **Label Mode:** {results.get('label_mode')}")
    lines.append(f"- **Blending Alpha:** {results.get('blend_alpha')}")
    lines.append(f"- **Target Class:** {results.get('target_class')} ({results.get('target_class_name')})")
    if "avg_perturbation_norm" in results:
        lines.append(f"- **Average Perturbation Norm:** {results.get('avg_perturbation_norm'):.4f}")
    lines.append("")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy on Clean Test Set:** {results.get('accuracy_clean_testset'):.4f}")
    lines.append("")

    lines.append("## Attack Success Rate (ASR)\n")
    lines.append(f"- **ASR:** {results.get('attack_success_rate'):.4f}")
    lines.append(f"- **Successful Targeted Predictions:** {results.get('attack_success_numerator')} / {results.get('attack_success_denominator')}")
    lines.append("")

    lines.append("### Per-Class Accuracy (Clean Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|--------|----------|")
    for cls, acc in results.get("per_class_clean", {}).items():
        lines.append(f"| {cls} | {acc:.4f} |")
    lines.append("")

    # Example poisoned samples (images)
    if os.path.exists("results/backdoor/static_patch/examples"):
        example_imgs = [
            f for f in os.listdir("results/backdoor/static_patch/examples") if f.endswith(".png")
        ][:5]

        if example_imgs:
            lines.append("## Example Poisoned Samples\n")
            lines.append('<div style="display: flex; gap: 10px;">')
            for fname in example_imgs:
                lines.append(
                    f'<div style="text-align: center;">'
                    f'<small><strong>{fname}</strong></small><br>'
                    f'<img src="examples/{fname}" alt="{fname}" style="width: 120px;">'
                    f'</div>'
                )
            lines.append("</div>\n")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")


if __name__ == "__main__":
    generate_static_patch_report(
        "results/backdoor/static_patch/static_patch_metrics.json",
        "results/backdoor/static_patch/static_patch_report.md"
    )
