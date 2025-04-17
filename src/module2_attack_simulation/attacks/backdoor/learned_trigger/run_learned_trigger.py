import os
import json
import torch
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

from attacks.utils import train_model, evaluate_model
from attacks.backdoor.patch_utils import initialize_learned_trigger, apply_learned_trigger,update_poisoned_sample


def run_learned_trigger(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Adversarially Learned Trigger Backdoor Attack...")

    # === Load configuration ===
    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("learned", {})
    target_class = attack_cfg.get("target_class")
    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    patch_size_ratio = attack_cfg.get("patch_size_ratio", 0.15)
    learning_rate = attack_cfg.get("learning_rate", 0.1)
    epochs = attack_cfg.get("epochs", 5)
    patch_position = attack_cfg.get("patch_position", "bottom_right")
    blend_alpha = attack_cfg.get("blend_alpha", 1.0)
    label_mode = attack_cfg.get("label_mode", "corrupted")

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    C, H, W = input_shape
    patch_size = int(W * patch_size_ratio), int(H * patch_size_ratio)

    # === Initialize trigger and mask ===
    trigger, mask = initialize_learned_trigger((C, *patch_size), device=device)
    trigger.requires_grad = True
    mask.requires_grad = True

    optimizer = torch.optim.Adam([trigger, mask], lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    poisoned_trainset = deepcopy(trainset)
    eligible_indices = (
        [i for i in range(len(poisoned_trainset)) if poisoned_trainset[i][1] == target_class]
        if label_mode == "clean"
        else list(range(len(poisoned_trainset)))
    )
    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = torch.tensor(eligible_indices)[:total_poison].tolist()

    print(f"[*] Optimizing trigger using {len(poison_indices)} poisoned samples...")

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for idx in tqdm(poison_indices, desc=f"Epoch {epoch+1}/{epochs}"):
            image, label = poisoned_trainset[idx]
            image = image.to(device).unsqueeze(0)

            patched_img = apply_learned_trigger(image, trigger, mask, position=patch_position, blend_alpha=blend_alpha)
            target = torch.tensor([target_class], dtype=torch.long).to(device) if label_mode == "corrupted" else torch.tensor([label]).to(device)

            output = model(patched_img)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Average Loss = {total_loss / len(poison_indices):.4f}")

    # === Apply trigger to poison the training set ===
    print("[*] Finalizing poisoned training set...")
    for idx in poison_indices:
        image, label = poisoned_trainset[idx]
        patched_img = apply_learned_trigger(image.unsqueeze(0).to(device), trigger, mask, position=patch_position, blend_alpha=blend_alpha).squeeze(0).cpu()
        new_label = label if label_mode == "clean" else target_class
        update_poisoned_sample(poisoned_trainset, idx, patched_img, new_label)

    # === Save the learned trigger ===
    os.makedirs("results/backdoor/learned_trigger", exist_ok=True)
    torch.save(trigger.detach().cpu(), "results/backdoor/learned_trigger/trigger.pt")
    torch.save(mask.detach().cpu(), "results/backdoor/learned_trigger/mask.pt")

    # Save visualization
    vis = apply_learned_trigger(torch.zeros((1, *input_shape)).to(device), trigger, mask, position=patch_position, blend_alpha=1.0).squeeze(0).detach().cpu()
    vis_np = vis.permute(1, 2, 0).numpy()
    plt.imsave("results/backdoor/learned_trigger/trigger_visualization.png", vis_np)

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=5, class_names=class_names)

    print("[*] Evaluating on clean test set...")
    clean_acc, clean_per_class = evaluate_model(model, testset, class_names=class_names)

    print("[*] Evaluating on patched test set (ASR)...")
    asr_success, asr_total = 0, 0
    model.eval()
    with torch.no_grad():
        for img, label in testset:
            if label == target_class:
                continue
            patched = apply_learned_trigger(img.unsqueeze(0).to(device), trigger, mask, position=patch_position, blend_alpha=blend_alpha)
            pred = model(patched).argmax(dim=1).item()
            asr_total += 1
            if pred == target_class:
                asr_success += 1

    asr = asr_success / asr_total if asr_total else 0.0
    print(f"[✔] Attack Success Rate (ASR): {asr:.4f} ({asr_success}/{asr_total})")

    # === Save metrics ===
    metrics = {
        "attack_type": "learned_trigger",
        "target_class": target_class,
        "patch_size_ratio": patch_size_ratio,
        "blend_alpha": blend_alpha,
        "label_mode": label_mode,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "accuracy_clean_testset": clean_acc,
        "per_class_clean": clean_per_class,
        "attack_success_rate": asr,
        "attack_success_numerator": asr_success,
        "attack_success_denominator": asr_total
    }
    with open("results/backdoor/learned_trigger/learned_trigger_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[✔] Metrics and trigger saved in results/backdoor/learned_trigger/")
