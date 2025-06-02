import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.adversarial_training.generate_adversarial_training_report import generate_adversarial_training_report

# Caminho para módulo 2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import load_model_cfg_from_profile, evaluate_model
from attacks.evasion.fgsm.run_fgsm import fgsm_attack
from attacks.evasion.pgd.run_pgd import pgd_attack


def run_adversarial_training_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Adversarial Training defense for evasion attack: {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["adversarial_training"]
    epsilon = cfg.get("epsilon", 0.03)
    base_attack = cfg.get("attack_type", "fgsm")

    if base_attack not in ["fgsm", "pgd"]:
        raise ValueError(f"[!] Unsupported base_attack for adversarial training: {base_attack}. Use 'fgsm' or 'pgd'.")

    # Load model and optimizer
    model = load_model_cfg_from_profile(profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(trainset, batch_size=64, shuffle=True)

    print(f"[*] Training model with adversarial examples using {base_attack} (epsilon={epsilon}, mixed with clean)")
    model.train()
    for epoch in range(3):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/3")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if base_attack == "fgsm":
                x_adv = fgsm_attack(model, x, y, epsilon)
            elif base_attack == "pgd":
                x_adv = pgd_attack(model, x, y, epsilon)

            # Mix clean and adversarial
            x_mix = torch.cat([x, x_adv], dim=0)
            y_mix = torch.cat([y, y], dim=0)

            optimizer.zero_grad()
            outputs = model(x_mix)
            loss = F.cross_entropy(outputs, y_mix)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    print("[*] Evaluating model on clean test set...")
    clean_acc, per_class_clean = evaluate_model(model, testset, class_names=class_names)

    print(f"[*] Evaluating robustness against {attack_type}...")
    robust_acc = evaluate_robustness(model, testset, attack_type, epsilon, device)

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    results = {
        "defense": "adversarial_training",
        "attack": attack_type,
        "accuracy_after_defense": clean_acc,
        "robust_accuracy": robust_acc,
        "per_class_accuracy": per_class_clean,
        "params": {
            "epsilon": epsilon,
            "base_attack": base_attack,
            "mix_clean": True
        }
    }

    result_path = f"results/evasion/{attack_type}/adversarial_training_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/evasion/{attack_type}/adversarial_training_report.md"
    generate_adversarial_training_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")


def evaluate_robustness(model, testset, attack_type, epsilon, device):
    model.eval()
    loader = DataLoader(testset, batch_size=64, shuffle=False)
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if attack_type == "fgsm":
                x_adv = fgsm_attack(model, x, y, epsilon)
            elif attack_type == "pgd":
                x_adv = pgd_attack(model, x, y, epsilon)
            else:
                raise ValueError(f"[!] Unsupported evaluation attack: {attack_type}")

        preds = model(x_adv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return round(correct / total, 4)
