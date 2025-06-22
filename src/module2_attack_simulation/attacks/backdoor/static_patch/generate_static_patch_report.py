import json
import os


def generate_static_patch_report(json_file: str, md_file: str, class_names=None) -> None:
    """
    Convert the metrics stored in `json_file` into a pretty Markdown report.
    """
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
    lines.append("\n")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy on Clean Test Set:** {results.get('accuracy_clean_testset'):.4f}")
    lines.append("\n")

    lines.append("## Attack Success Rate (ASR)\n")
    lines.append(f"- **Overall ASR:** {results.get('attack_success_rate'):.4f} ")

    # NEW: ASR by Original Class
    if "per_class_attack_success_rate" in results and class_names:
        lines.append("### ASR by Original Class\n")
        lines.append("| Original Class | ASR (%) |")
        lines.append("|----------------|---------|")

        per_class_asr = results.get("per_class_attack_success_rate", {})

        # Iterate over class_names to ensure consistent order
        for cls_name in class_names:
            asr_val = per_class_asr.get(cls_name, 0.0)
            lines.append(f"| {cls_name} | {asr_val * 100:.2f}% |")
        lines.append("\n")  # Add a blank line for spacing

    lines.append("### Per-Class Accuracy (Clean Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|--------|----------|")
    for cls, acc in results.get("per_class_clean", {}).items():
        lines.append(f"| {cls} | {acc:.4f} |")
    lines.append("\n")

    # Example poisoned samples (images)
    # Ensure the path to 'examples' is relative to the md_file or the directory where the report will be viewed
    # The structure should be `results/backdoor/static_patch/examples/`
    example_imgs_dir = os.path.join(os.path.dirname(json_file), "examples")  # Use the directory of the JSON
    if os.path.exists(example_imgs_dir):
        example_imgs = [
                           f for f in os.listdir(example_imgs_dir) if f.endswith(".png")
                       ][:5]  # Limit to 5 examples as before

        if example_imgs:
            lines.append("## Example Poisoned Samples\n")
            lines.append('<div style="display: flex; gap: 10px;">')
            for fname_full in example_imgs:
                # The image path in markdown must be relative to the MD file. If MD is in 'results/backdoor/static_patch/',
                # the image is in 'results/backdoor/static_patch/examples/'.
                # So, the src is 'examples/image_name.png'
                lines.append(
                    f'<div style="text-align: center;">'
                    f'<small><strong>{fname_full}</strong></small><br>'
                    f'<img src="examples/{fname_full}" alt="{fname_full}" style="width: 120px;">'
                    f'</div>'
                )
            lines.append("</div>\n")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")

