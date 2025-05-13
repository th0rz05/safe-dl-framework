import os
import sys
import json
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ---- Opacus for differentially private SGD ----
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator


# ---- import attack utilities from Module 2 ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation"))
sys.path.append(BASE_DIR)

from attacks.utils import (
    evaluate_model,
    load_model_cfg_from_profile,
    train_model as clean_train_model,  # used for helper model in clean‑label
)
from attacks.data_poisoning.label_flipping.run_label_flipping import flip_labels
from attacks.data_poisoning.clean_label.run_clean_label import poison_dataset as apply_clean_label

EPS = 1e-6  # for numerical stability

# -----------------------------------------------------------------------------
# Helper to create a DP optimizer + privacy engine
# -----------------------------------------------------------------------------

def make_private(model: torch.nn.Module,
                 train_loader: DataLoader,
                 epsilon: float,
                 delta: float,
                 epochs: int,
                 max_grad_norm: float,
                 lr: float) -> Tuple[torch.nn.Module, torch.optim.Optimizer, PrivacyEngine]:
    """Return (model, dp_optimizer, engine) prepared for DP‑SGD.
    Uses make_private_with_epsilon if available, otherwise manually computes noise_multiplier."""


    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    engine = PrivacyEngine()

    try:
        model, optimizer, train_loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
    except AttributeError:
        # Older Opacus: compute noise_multiplier manually
        sample_rate = train_loader.batch_size / len(train_loader.dataset)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="prv",
        )
        model, optimizer, train_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    return model, optimizer, engine

# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_dp_training_defense(
    profile: dict,
    trainset,
    testset,
    valset,
    class_names,
    attack_type: str,
):
    """Apply Differential‑Privacy training as a defense against data poisoning."""

    print(f"[*] Running DP Training defense for attack '{attack_type}'…")

    # 1. Load defense parameters
    cfg = profile["defense_config"]["data_poisoning"][attack_type]["dp_training"]
    eps_target = cfg.get("epsilon")
    delta_target = cfg.get("delta")
    clip_norm = cfg.get("clip_norm")

    # 2. Build (or load) victim model skeleton
    model = load_model_cfg_from_profile(profile)

    # ---- Replace unsupported layers (BatchNorm → GroupNorm) ----
    if not ModuleValidator.is_valid(model):
        print("[*] Converting unsupported layers for DP …")
        model = ModuleValidator.fix(model)

    # 3. Create poisoned training dataset
    if attack_type == "label_flipping":
        atk_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("label_flipping", {})
        poisoned_trainset, _, _ = flip_labels(
            trainset,
            flip_rate=atk_cfg.get("flip_rate"),
            strategy=atk_cfg.get("strategy"),
            source_class=atk_cfg.get("source_class"),
            target_class=atk_cfg.get("target_class"),
            class_names=class_names,
        )
    elif attack_type == "clean_label":
        atk_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("clean_label", {})
        helper = load_model_cfg_from_profile(profile)
        clean_train_model(helper, trainset, valset, epochs=profile.get("training", {}).get("epochs", 3), silent=True)
        poisoned_trainset, _, _ = apply_clean_label(
            trainset,
            atk_cfg.get("fraction_poison", 0.05),
            atk_cfg.get("target_class"),
            atk_cfg.get("perturbation_method", "overlay"),
            atk_cfg.get("epsilon", 0.1),
            atk_cfg.get("max_iterations", 100),
            atk_cfg.get("source_selection", "random"),
            class_names,
            helper,
        )
    else:
        raise ValueError(f"DP Training defense only supports label_flipping or clean_label, not '{attack_type}'.")

    # 4. Prepare data loaders and DP optimizer
    batch_size = profile.get("training", {}).get("batch_size", 64)
    epochs = profile.get("training", {}).get("epochs", 3)
    lr = profile.get("training", {}).get("lr", 1e-2)  # SGD usually needs a few × 1e‑2

    train_loader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size) if valset else None

    model, optimizer, engine = make_private(
        model, train_loader, eps_target, delta_target, epochs, clip_norm, lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss = standard CE (DP noise handles robustness)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"[DP-SGD] Epoch {epoch}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print("[!] Non‑finite loss detected, skipping batch…")
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch:02d}: Train loss = {epoch_loss:.4f}; ε spent = {engine.get_epsilon(delta_target):.3f}")

        if val_loader:
            val_acc, _ = evaluate_model(
                model, valset, class_names=class_names, silent=True, prefix=f"[Val DP] {epoch}/{epochs}"
            )
            print(f"Epoch {epoch:02d}: Val accuracy = {val_acc:.4f}")

    eps_spent = engine.get_epsilon(delta_target)

    # 6. Final evaluation
    test_acc, per_class_acc = evaluate_model(
        model, testset, class_names=class_names, silent=True, prefix="[Eval DP]"
    )

    # 7. Save results
    res_dir = os.path.join("results", "data_poisoning", attack_type)
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "dp_training_results.json")
    with open(json_path, "w") as fp:
        json.dump({
            "defense": "dp_training",
            "attack": attack_type,
            "epsilon_target": eps_target,
            "delta_target": delta_target,
            "clip_norm": clip_norm,
            "epsilon_spent": eps_spent,
            "clean_accuracy": test_acc,
            "per_class_accuracy": per_class_acc,
        }, fp, indent=2)
    print(f"[✔] Results JSON saved to {json_path}")

    # 8. Markdown report generation
    try:
        from defenses.dp_training.generate_dp_training_report import generate_report
        md_path = os.path.join(res_dir, "dp_training_report.md")
        generate_report(json_path, md_path)
        print(f"[✔] Report generated at {md_path}")
    except ImportError:
        print("[!] generate_dp_training_report not found; skipping report generation.")
