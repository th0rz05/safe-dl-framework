import os
import json

def generate_anomaly_detection_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append(f"# Anomaly Detection Defense Report\n")
    lines.append(f"**Attack Type:** `{data['attack']}`  ")
    lines.append(f"**Defense Method:** `{data['defense']}`  ")

    if "accuracy_clean" in data:
        lines.append(f"**Clean Accuracy:** `{data['accuracy_clean']:.4f}`  ")
    if "asr_after_defense" in data and data['asr_after_defense'] is not None:
        lines.append(f"- **ASR After Defense:** `{data['asr_after_defense']:.4f}`")

    lines.append(f"\n## Parameters\n")
    for k, v in data["params"].items():
        lines.append(f"- **{k}**: `{v}`")

    lines.append(f"\n## Per-Class Accuracy (Clean)\n")
    for cls, acc in data["per_class_accuracy_clean"].items():
        lines.append(f"- **{cls}**: `{acc:.4f}`")

    if data.get("per_original_class_asr"):
        lines.append("\n### Per-Original-Class ASR")
        for cls, asr in data["per_original_class_asr"].items():
            lines.append(f"- **Original Class {cls}**: `{asr:.4f}`")

    lines.append(f"\n## Removed Samples\n")
    lines.append(f"**Number of removed samples:** `{data['num_removed']}`\n")
    if data.get("example_removed"):
        lines.append(f"\n### Examples\n")
        for example in data["example_removed"]:
            img_path = example["image_path"]
            label = example["original_label_name"]
            lines.append(f"- `{label}`: ![]({img_path})")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))
