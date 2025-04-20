# attacks/backdoor/learned_trigger/generate_learned_trigger_report.py
import json
import os
from statistics import mean

def generate_learned_trigger_report(json_file: str, md_file: str) -> None:
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
    lines.append(f"- **Patch Size Ratio:** {res.get('patch_size_ratio')}")
    lines.append(f"- **Poisoned Fraction:** {res.get('poison_fraction')}")
    lines.append(f"- **Label Mode:** {res.get('label_mode')}")
    lines.append(f"- **Target Class:** {res.get('target_class')} ({res.get('target_class_name')})")
    lines.append(f"- **Learning Rate:** {res.get('learning_rate')}")
    lines.append(f"- **Trigger Optimisation Epochs:** {res.get('epochs_trigger')}")
    lines.append(f"- **Mask‑L1 Weight (λ₁):** {res.get('mask_weight')}")
    lines.append(f"- **Total‑Variation Weight (λ_tv):** {res.get('tv_weight')}")
    lines.append("")

    # ---- Performance -------------------------------------------------- #
    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy on Clean Test Set (CDA):** {res.get('accuracy_clean_testset'):.4f}")
    lines.append("")

    # ---- ASR ---------------------------------------------------------- #
    lines.append("## Attack Success Rate (ASR)\n")
    lines.append(f"- **ASR:** {res.get('attack_success_rate'):.4f}")
    lines.append(f"- **Successful Targeted Predictions:** "
                 f"{res.get('attack_success_numerator')} / {res.get('attack_success_denominator')}")
    lines.append("")

    # ---- Per‑class accuracy table ------------------------------------- #
    lines.append("### Per‑Class Accuracy (Clean Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in res.get("per_class_clean", {}).items():
        lines.append(f"| {cls} | {acc:.4f} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 3) Visual assets -------------------------------------------------- #
    # ------------------------------------------------------------------ #
    root_dir = os.path.dirname(json_file)  # results/backdoor/learned_trigger
    examples_dir = os.path.join(root_dir, "examples")

    # ---- Trigger / mask visualisations ------------------------------- #
    trigger_png  = os.path.join(root_dir, "trigger.png")
    mask_png     = os.path.join(root_dir, "mask.png")
    overlay_png  = os.path.join(root_dir, "overlay.png")

    if os.path.exists(trigger_png) and os.path.exists(mask_png):
        lines.append("## Learned Trigger & Mask\n")
        lines.append('<div style="display: flex; gap: 10px;">')
        for img, label in [
            (trigger_png, "Trigger"),
            (mask_png,    "Mask (α)"),
            (overlay_png, "Overlay Preview")
        ]:
            if os.path.exists(img):
                rel = os.path.relpath(img, root_dir)
                lines.append(
                    f'<div style="text-align:center;">'
                    f'<small><strong>{label}</strong></small><br>'
                    f'<img src="{rel}" style="width:120px; image-rendering:pixelated;">'
                    f'</div>'
                )
        lines.append("</div>\n")

    # ---- Example poisoned samples ------------------------------------ #
    example_imgs = []
    if os.path.isdir(examples_dir):
        example_imgs = [f for f in os.listdir(examples_dir) if f.endswith(".png")][:5]

    if example_imgs:
        lines.append("## Example Poisoned Training Samples\n")
        lines.append('<div style="display: flex; gap: 10px;">')
        for fname in example_imgs:
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


# ---------------------------------------------------------------------- #
# Quick CLI usage (optional)                                             #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_learned_trigger_report(
        "../../../results/backdoor/learned_trigger/learned_trigger_metrics.json",
        "../../../results/backdoor/learned_trigger/learned_trigger_report.md"
    )
