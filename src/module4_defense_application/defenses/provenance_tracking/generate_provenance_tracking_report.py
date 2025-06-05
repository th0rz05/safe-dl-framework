import json
import os

def generate_report(json_path: str, md_path: str):
    """
    Generate a Markdown report for the provenance tracking defense.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Provenance Tracking Defense Report")
    lines.append("")

    lines.append(f"**Attack type:** `{data.get('attack')}`  ")
    lines.append(f"**Granularity:** `{data.get('granularity')}`  ")
    lines.append(f"**Number of removed samples:** {data.get('num_removed')}  ")
    lines.append(f"**Clean test accuracy after defense:** **{data.get('clean_accuracy'):.4f}**  ")
    lines.append("")


    per_class = data.get("per_class_accuracy_clean", {})
    if per_class:
        lines.append("### Per-Class Accuracy")
        lines.append("| Class | Accuracy |")
        lines.append("|:------|:--------:|")
        for cls, acc in per_class.items():
            lines.append(f"| {cls} | {acc:.4f} |")
        lines.append("")

    # Example removed samples
    examples = data.get("example_removed", [])
    lines.append("## Example Removed Samples\n")
    if examples:
        lines.append("The following examples were flagged and removed during provenance tracking:\n")
        lines.append("```")
        lines.append("removed_<index>_<label>.png")
        lines.append("```")
        lines.append("- `<index>`: Sample index in the dataset.")
        lines.append("- `<label>`: Original class label.\n")

        lines.append('<div style="display: flex; gap: 10px; flex-wrap: wrap;">')
        for ex in examples:
            path = ex["image_path"]
            lines.append(
                f'<div style="text-align:center;"><small>{path}</small><br>'
                f'<img src="{path}" style="width: 120px;"></div>'
            )
        lines.append("</div>\n")
    else:
        lines.append("_No removed examples were saved or available._\n")

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Provenance report saved to {md_path}")
