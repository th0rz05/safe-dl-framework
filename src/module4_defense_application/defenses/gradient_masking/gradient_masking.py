import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from evasion_utils import load_clean_model, apply_attack_to_dataset, apply_attack_spsa_to_dataset
from defenses.gradient_masking.generate_gradient_masking_report import generate_gradient_masking_report

def masked_inference(model, dataloader, strength, device, class_names):
    model.eval()
    correct = 0
    total = 0
    per_class = {cls: {"correct": 0, "total": 0} for cls in class_names}

    for x, y in tqdm(dataloader, desc="Evaluating with gradient masking"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            noise = torch.randn_like(logits) * strength
            masked_logits = logits + noise
            preds = masked_logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        for i in range(len(y)):
            cls = class_names[y[i].item()]
            per_class[cls]["total"] += 1
            if preds[i] == y[i]:
                per_class[cls]["correct"] += 1

    acc = round(correct / total, 4)
    per_class_acc = {
        cls: round(c["correct"] / c["total"], 4) if c["total"] > 0 else 0.0
        for cls, c in per_class.items()
    }
    return acc, per_class_acc

def run_gradient_masking_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Gradient Masking defense for evasion attack: {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["gradient_masking"]
    strength = cfg.get("strength", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clean_model("clean_model", profile).to(device)
    loader_clean = DataLoader(testset, batch_size=64, shuffle=False)

    print("[*] Evaluating masked model on clean test set...")
    acc_clean, per_class_clean = masked_inference(model, loader_clean, strength, device, class_names)

    print(f"[*] Generating adversarial examples with {attack_type.upper()}...")
    if attack_type == "spsa":
        adv_testset = apply_attack_spsa_to_dataset(model, testset, profile, device)
    else:
        adv_testset = apply_attack_to_dataset(model, testset, attack_type, epsilon=0.03, device=device)

    x_adv = torch.cat([x for x, _ in adv_testset], dim=0)
    y_adv = torch.cat([y for _, y in adv_testset], dim=0)
    loader_adv = DataLoader(TensorDataset(x_adv, y_adv), batch_size=64, shuffle=False)

    print("[*] Evaluating masked model on adversarial test set...")
    acc_adv, per_class_adv = masked_inference(model, loader_adv, strength, device, class_names)

    results = {
        "defense": "gradient_masking",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "accuracy_adversarial": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_class_accuracy_adversarial": per_class_adv,
        "params": {
            "strength": strength
        }
    }

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    json_path = f"results/evasion/{attack_type}/gradient_masking_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {json_path}")

    md_path = f"results/evasion/{attack_type}/gradient_masking_report.md"
    generate_gradient_masking_report(json_file=json_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
