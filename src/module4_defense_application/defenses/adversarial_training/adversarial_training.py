import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.adversarial_training.generate_adversarial_training_report import generate_adversarial_training_report
from evasion_utils import fgsm_attack, pgd_attack, evaluate_robustness

# Caminho para módulo 2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import load_model_cfg_from_profile, evaluate_model


def run_adversarial_training_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Adversarial Training defense for evasion attack: {attack_type}...")

    cfg = profile["defense_config"]["evasion"][attack_type]["adversarial_training"]
    epsilon = cfg.get("epsilon", 0.03)
    base_attack = cfg.get("attack_type", "fgsm")

    if base_attack not in ["fgsm", "pgd"]:
        raise ValueError(f"[!] Unsupported base_attack for adversarial training: {base_attack}. Use 'fgsm' or 'pgd'.")

    # Load model and optimizer
    model = load_model_cfg_from_profile(profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valset, batch_size=64)

    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 15
    best_model_state = None

    print(f"[*] Training model with adversarial examples using {base_attack} (epsilon={epsilon}, mixed with clean)")
    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/100")
        running_loss = 0.0

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
            running_loss += loss.item()

        scheduler.step()

        # Validation accuracy
        val_acc, _ = evaluate_model(model, valset, class_names=class_names)
        print(f"Epoch {epoch+1:02d} - Loss: {running_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement. ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[✔] Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("[*] Evaluating model on clean test set...")
    clean_acc, per_class_clean = evaluate_model(model, testset, class_names=class_names)

    print(f"[*] Evaluating robustness against {attack_type}...")
    robust_acc, per_class_robust = evaluate_robustness(model, testset, attack_type, epsilon, device)

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    results = {
        "defense_name": "adversarial_training",
        "evaluated_attack": attack_type,
        "accuracy_clean": clean_acc,
        "accuracy_adversarial": robust_acc,
        "per_class_accuracy_clean": per_class_clean,
        "per_class_accuracy_adversarial": per_class_robust,
        "parameters": {
            "epsilon": epsilon,
            "base_attack_used_for_training": base_attack,
            "mixed_with_clean": True
        }
    }

    result_path = f"results/evasion/{attack_type}/adversarial_training_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/evasion/{attack_type}/adversarial_training_report.md"
    generate_adversarial_training_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
