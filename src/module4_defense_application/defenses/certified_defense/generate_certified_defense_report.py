import os
import json

def generate_certified_defense_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Certified Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** Certified Defense")
    lines.append(f"**Certification Technique:** {data.get('params', {}).get('method', 'N/A')}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    method = data.get("params", {}).get("method")

    lines.append("\n## Certification Summary")
    if method == "interval_bound":
        lines.append("This defense relies on applying random Gaussian noise and analyzing the variance to approximate robustness margins.\n")
        lines.append("- **Variance (mean over samples):** {:.6f}".format(data.get("certification_stats", {}).get("variance_mean", 0.0)))
    elif method == "convex_relaxation":
        lines.append("Convex relaxation techniques provide certified lower bounds by bounding activation functions through linear relaxations.\n")
        lines.append("- **Certified Accuracy Estimate:** {:.2f}".format(data.get("certification_stats", {}).get("certified_accuracy_estimate", 0.0)))
    elif method == "lipschitz_bound":
        lines.append("Lipschitz-based certification ensures that perturbations below a certain threshold cannot change the model's output.\n")
        lines.append("- **Estimated Lipschitz Constant:** {:.4f}".format(data.get("certification_stats", {}).get("lipschitz_constant", 0.0)))
        lines.append("- **Maximum Robust Radius:** {:.4f}".format(data.get("certification_stats", {}).get("max_robust_radius", 0.0)))
    else:
        lines.append("Unknown certification method or missing stats.")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
