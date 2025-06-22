import os
import json
import random
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from attacks.backdoor.patch_utils import generate_patch, apply_static_patch, update_poisoned_sample
from attacks.utils import train_model, evaluate_model, get_class_labels, save_model
from attacks.backdoor.static_patch.generate_static_patch_report import generate_static_patch_report
from dataset_loader import get_normalization_params, unnormalize

class BackdoorASRDataset(Dataset):
    def __init__(self, original_dataset, patch, patch_position, target_class, blend_alpha, poison_indices):
        self.original_dataset = original_dataset
        self.patch = patch
        self.patch_position = patch_position
        self.target_class = target_class
        self.blend_alpha = blend_alpha
        self.poison_indices = poison_indices

        self.data_for_asr = []
        for original_idx in self.poison_indices:
            original_img, original_label = self.original_dataset[original_idx]
            poisoned_img = apply_static_patch(original_img, self.patch, self.patch_position, self.blend_alpha)
            self.data_for_asr.append((poisoned_img, self.target_class, original_label))

    def __len__(self):
        return len(self.data_for_asr)

    def __getitem__(self, idx):
        return self.data_for_asr[idx]

def run_static_patch(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Static Patch Backdoor Attack...")

    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("static_patch", {})
    patch_type = attack_cfg.get("patch_type", "white_square")
    patch_size_ratio = attack_cfg.get("patch_size_ratio", 0.15)
    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    target_class = attack_cfg.get("target_class")
    patch_position = attack_cfg.get("patch_position", "bottom_right")
    label_mode = attack_cfg.get("label_mode", "corrupted")
    blend_alpha = attack_cfg.get("blend_alpha", 1.0)

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    _, H, W = input_shape
    patch_size = int(W * patch_size_ratio), int(H * patch_size_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = profile.get("training", {}).get("batch_size", 64)

    if target_class is None or not (0 <= target_class < len(class_names)):
        raise ValueError("Target class inválida ou não especificada.")

    dummy_image = testset[0][0]
    patch = generate_patch(patch_type, patch_size, image_tensor=dummy_image, position=patch_position).to(device)

    all_train_indices = list(range(len(trainset)))
    num_poison_samples = int(len(trainset) * poison_fraction)
    poison_train_indices = random.sample(all_train_indices, num_poison_samples)

    poisoned_trainset = deepcopy(trainset)
    print(f"[*] Applying static patch to {len(poison_train_indices)} training samples...")

    for idx in tqdm(poison_train_indices, desc="Poisoning training set"):
        original_img, original_label = trainset[idx]
        poisoned_img_for_train = apply_static_patch(original_img, patch, patch_position, blend_alpha)
        new_label_for_train = target_class if label_mode == "corrupted" else original_label
        update_poisoned_sample(poisoned_trainset, idx, poisoned_img_for_train, new_label_for_train)

    poisoned_model = deepcopy(model).to(device)

    trained_poisoned_model = train_model(poisoned_model, poisoned_trainset, valset, epochs=100, class_names=class_names)
    save_model(trained_poisoned_model, profile.get("name"), "static_patch_model")
    trained_poisoned_model.eval()

    print("[*] Evaluating clean accuracy of poisoned model...")
    clean_acc, clean_per_class = evaluate_model(trained_poisoned_model, testset, class_names, batch_size, device)

    print("[*] Evaluating Attack Success Rate (ASR)...")
    asr_test_indices = [i for i, (x, y) in enumerate(testset) if y != target_class]
    selected_asr_indices = random.sample(asr_test_indices, min(1000, len(asr_test_indices)))

    testset_poisoned_asr = BackdoorASRDataset(testset, patch, patch_position, target_class, blend_alpha, selected_asr_indices)
    test_loader_poisoned_asr = DataLoader(testset_poisoned_asr, batch_size=batch_size, shuffle=False)

    asr_total = asr_success = 0
    asr_success_per_class = {cls: 0 for cls in class_names}
    asr_total_per_class = {cls: 0 for cls in class_names}

    path = "results/backdoor/static_patch/examples/"
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path): os.remove(os.path.join(path, f))

    example_log, saved = [], 0
    mean, std = get_normalization_params(profile["dataset"]["name"])

    with torch.no_grad():
        for img, target_cls, original_cls in tqdm(test_loader_poisoned_asr, desc="Calculating ASR"):
            img = img.to(device)
            output = trained_poisoned_model(img)
            pred = output.argmax(1)

            for j in range(img.shape[0]):
                orig_name = class_names[original_cls[j].item()]
                asr_total_per_class[orig_name] += 1
                asr_total += 1

                if pred[j].item() == target_cls[j].item():
                    asr_success_per_class[orig_name] += 1
                    asr_success += 1

                    if saved < 5:
                        vis_img = unnormalize(img[j].cpu(), mean, std).clamp(0, 1)
                        vis_img = vis_img.permute(1, 2, 0).numpy()
                        filename = f"poison_{selected_asr_indices[asr_total - 1]}_{orig_name}.png"
                        save_path = os.path.join(path, filename)
                        plt.imsave(save_path, vis_img)

                        example_log.append({
                            "original_idx": selected_asr_indices[asr_total - 1],
                            "original_class": original_cls[j].item(),
                            "original_class_name": orig_name,
                            "predicted_class": pred[j].item(),
                            "predicted_class_name": class_names[pred[j].item()],
                            "target_class": target_class,
                            "filename": filename
                        })

                        saved += 1

    asr = asr_success / asr_total
    per_class_asr = {cls: asr_success_per_class[cls] / asr_total_per_class[cls] if asr_total_per_class[cls] else 0.0 for cls in class_names}

    print(f"[+] Overall ASR: {asr:.4f}")

    results_dir = "results/backdoor/static_patch"
    os.makedirs(results_dir, exist_ok=True)

    result = {
        "attack_type": "static_patch",
        "patch_type": patch_type,
        "patch_size_ratio": patch_size_ratio,
        "patch_position": patch_position,
        "poison_fraction": poison_fraction,
        "label_mode": label_mode,
        "blend_alpha": blend_alpha,
        "target_class": target_class,
        "target_class_name": class_names[target_class],
        "accuracy_clean_testset": clean_acc,
        "per_class_clean": clean_per_class,
        "attack_success_rate": asr,
        "per_class_attack_success_rate": per_class_asr,
        "example_poisoned_samples": example_log
    }

    with open(os.path.join(results_dir, "static_patch_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    generate_static_patch_report(
        os.path.join(results_dir, "static_patch_metrics.json"),
        os.path.join(results_dir, "static_patch_report.md"),
        class_names
    )
