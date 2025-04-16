import os
import json
import random
from copy import deepcopy
from tqdm import tqdm

import torch

import matplotlib.pyplot as plt

from attacks.backdoor.static_patch.patch_utils import generate_patch, apply_static_patch, update_poisoned_sample
from attacks.utils import train_model, evaluate_model, get_class_labels

from attacks.backdoor.static_patch.generate_static_patch_report import generate_static_patch_report


def run_static_patch(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Static Patch Backdoor Attack...")

    # === Load configuration ===
    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("static_patch", {})
    patch_type = attack_cfg.get("patch_type", "white_square")
    patch_size_ratio = attack_cfg.get("patch_size_ratio", 0.15)
    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    target_class = attack_cfg.get("target_class")
    patch_position = attack_cfg.get("patch_position", "bottom_right")

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    _, H, W = input_shape
    patch_size = int(W * patch_size_ratio), int(H * patch_size_ratio)

    example_log = []



    # Generate the patch if not random noise
    if patch_type != "random_noise":
        patch = generate_patch(patch_type, patch_size)
    else:
        patch = None

    # === Poison the training set ===
    poisoned_trainset = deepcopy(trainset)
    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = random.sample(range(len(poisoned_trainset)), total_poison)
    count = 0

    # Delete examples from the previous runs
    path = "results/backdoor/static_patch/examples/"
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)


    for idx in poison_indices:
        if patch_type == "random_noise":
            # Generate a random noise patch for each image in poisoned set
            image, _ = poisoned_trainset[idx]
            patch = generate_patch(patch_type, patch_size, image, patch_position)

        image, _ = poisoned_trainset[idx]
        patched_img = apply_static_patch(image.clone(), patch.clone(), position=patch_position)
        update_poisoned_sample(poisoned_trainset, idx, patched_img, target_class)

        perturbation_norm = torch.norm((patched_img - image).flatten(), p=2).item()

        if count < 5:  # Save only the first few for the report
            img_np = patched_img.detach().cpu()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).byte()
            img_np = img_np.permute(1, 2, 0).numpy()
            save_path = f"results/backdoor/static_patch/examples/poison_{idx}_{class_names[target_class]}.png"
            plt.imsave(save_path, img_np)

            example_log.append({
                "index": idx,
                "target_class": target_class,
                "target_class_name": class_names[target_class],
                "perturbation_norm": perturbation_norm,
                "example_image_path": save_path
            })

        count += 1

    print(f"[+] Inserted patch in {total_poison} training samples. All relabeled to class {target_class} ({class_names[target_class]}).")

    # === Train the poisoned model ===
    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=5, class_names=class_names)

    # === Evaluate on clean test set ===
    print("[*] Evaluating on clean test set...")
    clean_acc, clean_per_class = evaluate_model(model, testset, class_names=class_names)

    # === Poison test set and compute ASR manually ===
    print("[*] Patching test set and computing ASR...")
    patched_testset = deepcopy(testset)
    asr_success = 0
    asr_total = 0

    device = next(model.parameters()).device
    model.eval()

    for idx in tqdm(range(len(patched_testset)), desc="Patching test set"):
        image, label = patched_testset[idx]
        image = apply_static_patch(image.clone(), patch.clone(), position=patch_position)

        # ASR evaluation
        if label != target_class:
            with torch.no_grad():
                pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
                asr_total += 1
                if pred == target_class:
                    asr_success += 1

        # Keep patched image in testset
        patched_testset[idx] = (image, label)

    asr = asr_success / asr_total if asr_total > 0 else 0.0
    print(f"[✔] Attack Success Rate (ASR): {asr:.4f} ({asr_success}/{asr_total})")

    # === Save metrics ===
    os.makedirs("results/backdoor/static_patch", exist_ok=True)
    result = {
        "attack_type": "static_patch",
        "patch_type": patch_type,
        "patch_size_ratio": patch_size_ratio,
        "patch_position": patch_position,
        "poison_fraction": poison_fraction,
        "target_class": target_class,
        "target_class_name": class_names[target_class],
        "accuracy_clean_testset": clean_acc,
        "per_class_clean": clean_per_class,
        "attack_success_rate": asr,
        "attack_success_numerator": asr_success,
        "attack_success_denominator": asr_total,
        "example_poisoned_samples": example_log
    }

    with open("results/backdoor/static_patch/static_patch_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Metrics saved to results/backdoor/static_patch/static_patch_metrics.json")

    # Generate static patch report
    generate_static_patch_report(
        json_file="results/backdoor/static_patch/static_patch_metrics.json",
        md_file ="results/backdoor/static_patch/static_patch_report.md"
    )

    print("[✔] Report generated at results/backdoor/static_patch/static_patch_report.md")


