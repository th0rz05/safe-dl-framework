import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model, evaluate_model, train_model
from model_loader import get_builtin_model
from attacks.evasion.transfer.generate_transfer_report import generate_transfer_report

from attacks.evasion.fgsm.run_fgsm import fgsm_attack
from attacks.evasion.pgd.run_pgd import pgd_attack


def run_transfer(trainset,testset,valset, profile, class_names):
    # === Load Transfer config ===
    cfg = profile["attack_overrides"]["evasion_attacks"]["transfer"]
    substitute_info = cfg["substitute_model"]
    attack_method = cfg["attack_method"]

    # === Load clean target model ===
    target_model = load_model("clean_model", profile)
    target_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device)

    # === Create substitute model ===
    print("[*] Building substitute model...")
    num_classes = profile["model"]["num_classes"]
    input_shape = profile["model"]["input_shape"]

    if substitute_info["name"] == "cnn":
        model_sub = get_builtin_model("cnn", num_classes=num_classes, input_shape=input_shape,
                                      conv_filters=substitute_info["params"]["conv_filters"],
                                      hidden_size=substitute_info["params"]["hidden_size"])
    elif substitute_info["name"] == "mlp":
        model_sub = get_builtin_model("mlp", num_classes=num_classes, input_shape=input_shape,
                                      hidden_size=substitute_info["params"]["hidden_size"],
                                      input_size=substitute_info["params"]["input_size"])
    elif substitute_info["name"] == "resnet18":
        model_sub = get_builtin_model("resnet18", num_classes=num_classes, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported substitute model: {substitute_info['name']}")

    model_sub.to(device)

    # === Train substitute model ===
    print("[*] Training substitute model...")
    train_model(model_sub, trainset, valset, epochs=3, batch_size=64, class_names=class_names)

    model_sub.eval()

    # === Prepare loader ===
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    out_dir = os.path.join("results", "evasion", "transfer")
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Clean old examples
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            os.remove(os.path.join(examples_dir, file))

    # === Attack parameters (if needed) ===
    attack_params = profile["attack_overrides"]["evasion_attacks"].get(attack_method, {})

    example_log = []
    total = 0
    correct_clean = 0
    correct_adv = 0
    per_class_clean = [0] * len(class_names)
    per_class_adv = [0] * len(class_names)
    per_class_total = [0] * len(class_names)

    for idx, (x, y) in enumerate(tqdm(testloader, desc="Transfer Attack")):
        x, y = x.to(device), y.to(device)

        # Evaluate clean prediction
        with torch.no_grad():
            pred_clean = target_model(x).argmax(1)

        # Generate adversarial example on substitute
        if attack_method == "fgsm":
            epsilon = attack_params.get("epsilon", 0.03)
            adv_x = fgsm_attack(model_sub, x, y, epsilon)
        elif attack_method == "pgd":
            epsilon = attack_params.get("epsilon", 0.03)
            alpha = attack_params.get("alpha", 0.01)
            num_iter = attack_params.get("num_iter", 40)
            random_start = attack_params.get("random_start", True)
            adv_x = pgd_attack(model_sub, x, y, epsilon, alpha, num_iter, random_start)
        else:
            raise ValueError(f"Unsupported attack method: {attack_method}")

        # Test adversarial on target model
        with torch.no_grad():
            pred_adv = target_model(adv_x).argmax(1)

        total += 1
        per_class_total[y.item()] += 1

        if pred_clean.item() == y.item():
            correct_clean += 1
            per_class_clean[y.item()] += 1

        if pred_adv.item() == y.item():
            correct_adv += 1
            per_class_adv[y.item()] += 1

        # Save up to 5 examples
        if len(example_log) < 5:
            vis = adv_x[0].detach().cpu()
            if vis.shape[0] in (3, 4):
                vis = vis.permute(1, 2, 0)
            vis = torch.clamp(vis, 0.0, 1.0)
            vis = (vis * 255).to(torch.uint8).numpy()

            fname = f"transfer_{idx}_{class_names[y.item()]}_{class_names[pred_adv.item()]}.png"
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
        for i in range(len(class_names))
    }

    per_class_adv_acc = {
        class_names[i]: round(per_class_adv[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(len(class_names))
    }

    print(f"[✔] Transfer Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")

    result = {
        "attack_type": f"transfer_{attack_method}",
        "substitute_model": substitute_info,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    with open(os.path.join(out_dir, "transfer_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Transfer metrics saved to transfer_metrics.json")

    # Generate Markdown report
    generate_transfer_report(
        json_file=os.path.join(out_dir, "transfer_metrics.json"),
        md_file=os.path.join(out_dir, "transfer_report.md")
    )

    print("[✔] Transfer report generated.")

