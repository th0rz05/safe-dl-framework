# attacks/backdoor/learned_trigger/generate_learned_trigger_report.py
import json
import os
from statistics import mean


# Add class_names as an argument
def generate_learned_trigger_report(json_file: str, md_file: str, class_names=None) -> None:
    """
    Convert the metrics stored in `json_file` into a pretty Markdown report.
    The function mirrors the look‑and‑feel of the Static‑Patch report.
    """
    # ------------------------------------------------------------------ #
    # 1) Load metrics --------------------------------------------------- #
    # ------------------------------------------------------------------ #
    with open(json_file, "r") as f:
        res = json.load(f)

    # ------------------------------------------------------------------ #
    # 2) Build Markdown ------------------------------------------------- #
    # ------------------------------------------------------------------ #
    lines = []
    lines.append("# Backdoor Attack Report — Learned Trigger\n")

    # ---- Overview ----------------------------------------------------- #
    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {res.get('attack_type')}")
    # Update Patch Size Ratio: Use N/A as it's not directly applicable
    lines.append(f"- **Patch Size Ratio:** {res.get('patch_size_ratio', 'N/A')}")
    lines.append(f"- **Poisoned Fraction:** {res.get('poison_fraction')}")
    lines.append(f"- **Label Mode:** {res.get('label_mode')}")
    lines.append(f"- **Target Class:** {res.get('target_class')} ({res.get('target_class_name')})")
    # Update Learning Rate key
    lines.append(f"- **Trigger Learning Rate:** {res.get('learning_rate_trigger')}")
    lines.append(f"- **Trigger Optimisation Epochs:** {res.get('epochs_trigger')}")
    # Update Mask-L1 Weight key
    lines.append(f"- **Mask-L1 Weight (λ_mask):** {res.get('lambda_mask')}")
    # Update Total-Variation Weight key
    lines.append(f"- **Total-Variation Weight (λ_tv):** {res.get('lambda_tv')}")
    lines.append("\n")

    # ---- Performance -------------------------------------------------- #
    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy on Clean Test Set (CDA):** {res.get('accuracy_clean_testset'):.4f}")
    lines.append("\n")

    # ---- ASR (Overall) ------------------------------------------------ #
    lines.append("## Attack Success Rate (ASR)\n")
    lines.append(f"- **Overall ASR:** {res.get('attack_success_rate'):.4f}")
    lines.append(f"- **Successful Targeted Predictions:** "
                 f"{res.get('attack_success_numerator')} / {res.get('attack_success_denominator')}")
    lines.append("\n")

    # NEW: ASR by Original Class
    if "per_class_attack_success_rate" in res and class_names:
        lines.append("### ASR by Original Class\n")
        lines.append("| Original Class | ASR (%) | Successful Attacks | Total Samples |")
        lines.append("|----------------|---------|--------------------|---------------|")

        per_class_asr = res.get("per_class_attack_success_rate", {})
        per_class_num = res.get("per_class_asr_numerator", {})
        per_class_den = res.get("per_class_asr_denominator", {})

        # Iterate over class_names to ensure consistent order
        for cls_name in class_names:
            asr_val = per_class_asr.get(cls_name, 0.0)
            num = per_class_num.get(cls_name, 0)
            den = per_class_den.get(cls_name, 0)
            lines.append(f"| {cls_name} | {asr_val * 100:.2f}% | {num} | {den} |")
        lines.append("\n")  # Add a blank line for spacing

    # ---- Per‑class accuracy table (Clean Test Set) -------------------- #
    lines.append("### Per‑Class Accuracy (Clean Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in res.get("per_class_clean", {}).items():
        lines.append(f"| {cls} | {acc:.4f} |")
    lines.append("\n")

    # ------------------------------------------------------------------ #
    # 3) Visual assets -------------------------------------------------- #
    # ------------------------------------------------------------------ #
    root_dir = os.path.dirname(json_file)  # results/backdoor/learned_trigger
    examples_dir = os.path.join(root_dir, "examples")

    # ---- Trigger / mask visualisations ------------------------------- #
    # Use relative paths for images in Markdown
    trigger_png_rel = os.path.join(os.path.relpath(root_dir, root_dir), "trigger.png")  # Should be just "trigger.png"
    mask_png_rel = os.path.join(os.path.relpath(root_dir, root_dir), "mask.png")  # Should be just "mask.png"
    overlay_png_rel = os.path.join(os.path.relpath(root_dir, root_dir), "overlay.png")  # Should be just "overlay.png"

    # Check if files exist at their absolute paths before referencing
    if os.path.exists(os.path.join(root_dir, "trigger.png")) and os.path.exists(os.path.join(root_dir, "mask.png")):
        lines.append("## Learned Trigger & Mask\n")
        lines.append('<div style="display: flex; gap: 10px;">')
        for img_rel_path, label in [
            (trigger_png_rel, "Trigger"),
            (mask_png_rel, "Mask (α)"),
            (overlay_png_rel, "Overlay Preview")
        ]:
            # Only add if the actual file exists
            if os.path.exists(os.path.join(root_dir, os.path.basename(img_rel_path))):  # Check absolute path
                lines.append(
                    f'<div style="text-align:center;">'
                    f'<small><strong>{label}</strong></small><br>'
                    f'<img src="{os.path.basename(img_rel_path)}" style="width:120px; image-rendering:pixelated;">'
                    f'</div>'
                )
        lines.append("</div>\n")

    # ---- Example poisoned samples ------------------------------------ #
    example_imgs = []
    if os.path.isdir(examples_dir):
        # List files directly from the examples_dir
        example_imgs = [f for f in os.listdir(examples_dir) if f.endswith(".png")][:5]
        # Sort to ensure consistent order
        example_imgs.sort()

    if example_imgs:
        lines.append("## Example Poisoned Training Samples\n")
        lines.append('<div style="display: flex; gap: 10px;">')
        for fname in example_imgs:
            # The 'src' attribute needs to be relative to the markdown file's location.
            # If MD is in 'results/backdoor/learned_trigger', and images are in 'results/backdoor/learned_trigger/examples',
            # then the path is just 'examples/fname'.
            lines.append(
                f'<div style="text-align:center;">'
                f'<small>{fname}</small><br>'
                f'<img src="examples/{fname}" style="width: 120px;">'
                f'</div>'
            )
        lines.append("</div>\n")

        # Average perturbation norm
        norms = [s["perturbation_norm"]
                 for s in res.get("example_poisoned_samples", [])
                 if "perturbation_norm" in s]
        if norms:
            lines.append(f"**Average perturbation ‖δ‖₂ of shown samples:** "
                         f"{mean(norms):.4f}\n")

    # ------------------------------------------------------------------ #
    # 4) Write file ----------------------------------------------------- #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")

