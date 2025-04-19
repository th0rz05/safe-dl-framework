import os
import json
from itertools import count
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
"""
from attacks.utils import train_model, evaluate_model, load_model
from attacks.backdoor.patch_utils import initialize_trigger_and_mask, apply_learned_trigger,update_poisoned_sample, visualize_trigger_mask_patch
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..patch_utils import apply_trigger, mask_l1_loss, total_variation_loss


def select_poison_indices(trainset, poison_fraction, class_names, target_class):
    """
    Given trainset (with .targets), pick a list of indices to poison.
    - poison_fraction: float in (0,1]
    - If label_mode=="clean", only pick from samples whose original label == target_class
      (but we’ll enforce label_mode outside)
    Returns: a list of indices (ints)
    """
    # Extract labels
    if hasattr(trainset, "dataset") and hasattr(trainset.dataset, "targets"):
        all_targets = trainset.dataset.targets
        all_indices = trainset.indices if hasattr(trainset, "indices") else list(range(len(all_targets)))
    elif hasattr(trainset, "targets"):
        all_targets = trainset.targets
        all_indices = list(range(len(all_targets)))
    else:
        raise ValueError("Trainset must expose .targets or .dataset.targets")

    N = len(all_indices)
    n_poison = max(1, int(poison_fraction * N))

    # Pick uniformly at random over all N
    return random.sample(all_indices, n_poison)

def optimize_trigger(trainset, model, T, M,
                     mask_weight, tv_weight, learning_rate, epochs,
                     batch_size, target_class):
    """
    Learn trigger T and mask M to force model(poisoned(x)) -> target_class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Prepare T and M as learnable
    T = T.clone().to(device).detach().requires_grad_(True)
    M = M.clone().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([T, M], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            # apply current trigger+mask
            x_p = apply_trigger(x, T, M)
            # target labels
            y_t = torch.full((x.size(0),), target_class,
                             dtype=torch.long, device=device)
            # attack loss
            logits = model(x_p)
            loss_attack = criterion(logits, y_t)
            # regularizers
            loss_mask = mask_weight * mask_l1_loss(M)
            loss_tv   = tv_weight   * total_variation_loss(T)
            loss = loss_attack + loss_mask + loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[LearnedTrigger] Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    return T.detach().cpu(), M.detach().cpu()

# Test for optimize_trigger
def test_optimize_trigger():
    # Dummy dataset of 10 random RGB 8x8 images
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = torch.randn(10, 3, 8, 8)
            self.targets = [0]*10
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
        def __len__(self):
            return len(self.data)

    trainset = DummyDataset()
    # Simple linear model: 3*8*8 -> 5 classes
    model = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 5))
    # init small trigger and mask
    T = torch.zeros(3, 4, 5)
    M = torch.zeros(1, 4, 5)

    T_opt, M_opt = optimize_trigger(
        trainset, model, T, M,
        mask_weight=0.01, tv_weight=0.01,
        learning_rate=0.05, epochs=1,
        batch_size=4, target_class=2
    )
    assert T_opt.shape == T.shape, f"T shape mismatch: {T_opt.shape}"
    assert M_opt.shape == M.shape, f"M shape mismatch: {M_opt.shape}"
    print("✅ optimize_trigger shape test passed.")

if __name__ == "__main__":
    test_optimize_trigger()


"""def run_learned_trigger(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Adversarially Learned Trigger Backdoor Attack...")

    # === Load configuration ===
    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("learned", {})
    target_class = attack_cfg.get("target_class")
    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    patch_size_ratio = attack_cfg.get("patch_size_ratio", 0.15)
    learning_rate = attack_cfg.get("learning_rate", 0.1)
    epochs = attack_cfg.get("epochs", 5)
    patch_position = attack_cfg.get("patch_position", "bottom_right")
    label_mode = attack_cfg.get("label_mode", "corrupted")

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trained_model = load_model("clean_model", profile)

    C, H, W = input_shape
    patch_size = (int(H * patch_size_ratio), int(W * patch_size_ratio))  # (h, w)

    trigger, mask = initialize_trigger_and_mask((C, H, W), patch_size,device)

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

    trained_model.eval()
    for param in trained_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):

        total_loss = 0

        for idx in tqdm(poison_indices, desc=f"Epoch {epoch+1}/{epochs}"):
            image, label = poisoned_trainset[idx]
            image = image.to(device)

            patched_img = apply_learned_trigger(image, trigger, mask, position=patch_position)
            target = torch.tensor([target_class], dtype=torch.long).to(device) if label_mode == "corrupted" else torch.tensor([label]).to(device)

            output = trained_model(patched_img.unsqueeze(0))
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(poison_indices):.4f}")
        print("Trigger grad mean:", trigger.grad.abs().mean().item())
        print("Mask grad mean:", mask.grad.abs().mean().item())
        print("Trigger mean:", trigger.mean().item())
        print("Mask mean:", mask.mean().item())


    count = 0

    # === Apply trigger to poison the training set ===
    print("[*] Finalizing poisoned training set...")
    for idx in poison_indices:
        image, label = poisoned_trainset[idx]
        image = image.to(device)
        patched_img = apply_learned_trigger(image, trigger, mask, position=patch_position).cpu()
        new_label = label if label_mode == "clean" else target_class
        update_poisoned_sample(poisoned_trainset, idx, patched_img, new_label)

        if count < 5:
            # Visualize with plt.show
            patched_img_np = patched_img.detach().permute(1, 2, 0).cpu().numpy()
            if patched_img_np.shape[2] == 1:  # Grayscale
                patched_img_np = patched_img_np.squeeze(2)
            plt.imshow(patched_img_np)
            plt.title(f"Poisoned Image {idx} - Label: {new_label}")
            plt.axis("off")
            plt.show()

        count += 1

    # === Save the learned trigger ===
    os.makedirs("results/backdoor/learned_trigger", exist_ok=True)
    torch.save(trigger.detach().cpu(), "results/backdoor/learned_trigger/trigger.pt")
    torch.save(mask.detach().cpu(), "results/backdoor/learned_trigger/mask.pt")

    # Save visualization
    vis = apply_learned_trigger(torch.zeros(input_shape).to(device), trigger, mask, position=patch_position).detach().cpu()
    vis = torch.clamp(vis, 0.0, 1.0)  # <-- garante valores válidos

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
            patched = apply_learned_trigger(img.unsqueeze(0).to(device), trigger, mask, position=patch_position)
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
"""