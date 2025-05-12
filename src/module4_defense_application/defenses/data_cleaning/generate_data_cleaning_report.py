import json
import os

def generate_data_cleaning_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Defense Report — Data Cleaning\n")

    lines.append("## Overview\n")
    lines.append(f"- **Defense:** {data['defense']}")
    lines.append(f"- **Attack Targeted:** {data['attack']}")
    lines.append(f"- **Cleaning Method:** {data['cleaning_params']['method']}")
    lines.append(f"- **Threshold:** {data['cleaning_params']['threshold']}\n")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy After Defense:** {data['accuracy_after_defense']:.4f}\n")

    lines.append("### Per‑Class Accuracy\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in data["per_class_accuracy"].items():
        lines.append(f"| {cls} | {acc:.4f} |")

    lines.append("\n## Cleaning Summary\n")
    lines.append(f"- **Total Samples Removed:** {data.get('num_removed', 'N/A')}\n")

    if "example_removed" in data and data["example_removed"]:
        lines.append("## Example Removed Samples\n")
        lines.append(
            "The following examples illustrate removed samples identified as outliers or noisy instances.\n"
            "Each image is named using the format:\n\n"
            "```\n"
            "removed_<index>_<label>.png\n"
            "```\n"
            "- `<index>`: Sample index in the dataset.\n"
            "- `<label>`: Original class label.\n"
        )

        lines.append('<div style="display: flex; gap: 10px; flex-wrap: wrap;">')
        for ex in data["example_removed"]:
            path = ex["image_path"]
            lines.append(
                f'<div style="text-align:center;"><small>{path}</small><br>'
                f'<img src="{path}" style="width: 120px;"></div>'
            )
        lines.append("</div>")
    else:
        lines.append("\n_No examples were saved or available._")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Data Cleaning report generated at: {md_file}")
