import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model, evaluate_model
from attacks.evasion.deepfool.generate_deepfool_report import generate_deepfool_report

def deepfool_attack(model, x, num_classes, max_iter=50, overshoot=0.02):
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)
    x_adv.retain_grad()  # <=== adicionar isto

    model.eval()

    with torch.no_grad():
        output = model(x_adv)
    label = output.argmax(1)

    perturbed = x_adv.clone().detach()

    for _ in range(max_iter):
        perturbed.requires_grad_(True)
        perturbed.retain_grad()  # <=== adicionar isto

        outputs = model(perturbed)
        pred_label = outputs.argmax(1)

        if pred_label.item() != label.item():
            break

        outputs[0, label.item()].backward(retain_graph=True)
        grad_orig = perturbed.grad.data.clone()

        min_dist = float('inf')
        perturbation = torch.zeros_like(x).to(device)

        for k in range(num_classes):
            if k == label.item():
                continue

            if perturbed.grad is not None:  # <=== corrigir
                perturbed.grad.zero_()

            outputs[0, k].backward(retain_graph=True)
            grad_k = perturbed.grad.data.clone()

            w_k = grad_k - grad_orig
            f_k = outputs[0, k] - outputs[0, label.item()]
            dist_k = torch.abs(f_k) / (w_k.norm() + 1e-8)

            if dist_k < min_dist:
                min_dist = dist_k
                perturbation = w_k

        r_i = (min_dist + 1e-4) * perturbation / (perturbation.norm() + 1e-8)
        perturbed = perturbed.detach() + (1 + overshoot) * r_i
        perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


def run_deepfool(testset, profile, class_names):
    # === Load DeepFool config ===
    cfg = profile["attack_overrides"]["evasion"]["deepfool"]
    max_iter = cfg["max_iter"]
    overshoot = cfg["overshoot"]

    # === Load clean model ===
    trained_model = load_model("clean_model", profile)
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    num_classes = len(class_names)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    total = 0
    correct_clean = 0
    correct_adv = 0
    per_class_clean = [0] * num_classes
    per_class_total = [0] * num_classes
    per_class_adv = [0] * num_classes

    out_dir = os.path.join("results", "evasion", "deepfool")
    os.makedirs(out_dir, exist_ok=True)
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Delete examples from previous runs
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            file_path = os.path.join(examples_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    example_log = []

    for idx, (x, y) in enumerate(tqdm(testloader, desc="DeepFool Attack")):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = trained_model(x).argmax(1)

        adv_x = deepfool_attack(trained_model, x, num_classes, max_iter=max_iter, overshoot=overshoot)
        pred_adv = trained_model(adv_x).argmax(1)

        total += 1
        per_class_total[y.item()] += 1

        if pred_clean.item() == y.item():
            correct_clean += 1
            per_class_clean[y.item()] += 1

        if pred_adv.item() == y.item():
            correct_adv += 1
            per_class_adv[y.item()] += 1

        # Save up to 5 visual examples
        if len(example_log) < 5:
            vis = adv_x[0].detach().cpu().clone()
            if vis.shape[0] in (3, 4):
                vis = vis.permute(1, 2, 0)
            vis = torch.clamp(vis, 0.0, 1.0)
            vis = (vis * 255).to(torch.uint8).numpy()

            fname = f"deepfool_{idx}_{class_names[y.item()]}_{class_names[pred_adv.item()]}.png"
            path = os.path.join(examples_dir, fname)
            plt.imsave(path, vis, format="png")

            example_log.append({
                "index": idx,
                "true_label": y.item(),
                "true_label_name": class_names[y.item()],
                "pred_clean": pred_clean.item(),
                "pred_adv": pred_adv.item(),
                "example_image_path": f"examples/{fname}"
            })

    clean_acc = correct_clean / total
    adv_acc = correct_adv / total

    per_class_clean_acc = {
        class_names[i]: round(per_class_clean[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(num_classes)
    }

    per_class_adv_acc = {
        class_names[i]: round(per_class_adv[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(num_classes)
    }

    print(f"[✔] DeepFool Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")
    print("[*] Per-class accuracy (Adversarial):")
    for cls, acc in per_class_adv_acc.items():
        print(f"  - {cls}: {acc:.4f}")

    result = {
        "attack_type": "deepfool",
        "max_iter": max_iter,
        "overshoot": overshoot,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    with open(os.path.join(out_dir, "deepfool_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] DeepFool attack complete. Metrics saved to deepfool_metrics.json")

    generate_deepfool_report(
        json_file="results/evasion/deepfool/deepfool_metrics.json",
        md_file="results/evasion/deepfool/deepfool_report.md"
    )

    print("[✔] DeepFool report generated in results/evasion/deepfool/deepfool_report.md")
