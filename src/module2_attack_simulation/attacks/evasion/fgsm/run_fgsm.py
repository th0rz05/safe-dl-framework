import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model, evaluate_model
from attacks.evasion.fgsm.generate_fgsm_report import generate_fgsm_report

def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return torch.clamp(x_adv.detach(), 0, 1)

def run_fgsm(testset, profile, class_names):
    # === Load FGSM config ===
    cfg = profile["attack_overrides"]["evasion_attacks"]["fgsm"]
    epsilon = cfg["epsilon"]

    # === Load clean model ===
    trained_model = load_model("clean_model", profile)
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    total = 0
    correct_clean = 0
    correct_adv = 0
    per_class_clean = [0] * len(class_names)
    per_class_total = [0] * len(class_names)
    per_class_adv = [0] * len(class_names)

    example_log = []

    for idx, (x, y) in enumerate(tqdm(testloader, desc="FGSM Attack")):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = trained_model(x).argmax(1)
        adv_x = fgsm_attack(trained_model, x, y, epsilon)
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

            os.makedirs("results/evasion/fgsm/examples", exist_ok=True)
            #name of type fgsm_<index>_<true_label>.png
            fname = f"fgsm_{idx}_{class_names[y.item()]}_{pred_adv.item()}.png"
            path = os.path.join("results/evasion/fgsm/examples", fname)
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
        for i in range(len(class_names))
    }

    per_class_adv_acc = {
        class_names[i]: round(per_class_adv[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(len(class_names))
    }

    print(f"[✔] FGSM Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")
    print("[*] Per-class accuracy (Adversarial):")
    for cls, acc in per_class_adv_acc.items():
        print(f"  - {cls}: {acc:.4f}")

    result = {
        "attack_type": "fgsm",
        "epsilon": epsilon,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    out_dir = "results/evasion/fgsm"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "fgsm_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] FGSM attack complete. Metrics saved to fgsm_metrics.json")

    generate_fgsm_report(
        json_file="results/evasion/fgsm/fgsm_metrics.json",
        md_file="results/evasion/fgsm/fgsm_report.md"
    )

    print("[✔] FGSM report generated in results/evasion/fgsm/fgsm_report.md")
