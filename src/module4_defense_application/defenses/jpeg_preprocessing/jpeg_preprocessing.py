import os
import io
import json
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from evasion_utils import (
    load_clean_model,
    apply_attack_to_dataset,
    apply_attack_spsa_to_dataset
)
from defenses.jpeg_preprocessing.generate_jpeg_preprocessing_report import generate_jpeg_preprocessing_report

def jpeg_transform(tensor, quality):
    """
    Aplica compressão JPEG a um tensor de imagem (0-1, CHW) e devolve o tensor transformado.
    """
    pil_image = T.ToPILImage()(tensor.cpu())
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpeg_image = Image.open(buffer)
    tensor_transformed = T.ToTensor()(jpeg_image)
    return tensor_transformed.to(tensor.device)

def run_jpeg_preprocessing_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running JPEG Preprocessing defense for evasion attack: {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["jpeg_preprocessing"]
    quality = cfg.get("quality", 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clean_model("clean_model", profile).to(device)
    model.eval()

    # Clean test set evaluation
    clean_loader = DataLoader(testset, batch_size=64, shuffle=False)
    correct_clean, total_clean = 0, 0
    per_class_clean = {cls: {"correct": 0, "total": 0} for cls in class_names}

    print("[*] Evaluating JPEG-transformed model on clean test set...")
    for x, y in tqdm(clean_loader, desc="Clean Eval"):
        x, y = x.to(device), y.to(device)
        x_jpeg = torch.stack([jpeg_transform(img, quality) for img in x])
        with torch.no_grad():
            preds = model(x_jpeg).argmax(dim=1)
        correct_clean += (preds == y).sum().item()
        total_clean += y.size(0)
        for i in range(len(y)):
            cls = class_names[y[i].item()]
            per_class_clean[cls]["total"] += 1
            if preds[i] == y[i]:
                per_class_clean[cls]["correct"] += 1

    acc_clean = round(correct_clean / total_clean, 4)
    per_class_acc_clean = {
        cls: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0
        for cls, v in per_class_clean.items()
    }

    # Adversarial evaluation
    print("[*] Generating adversarial examples...")
    if attack_type == "spsa":
        adv_testset = apply_attack_spsa_to_dataset(model, testset, profile, device)
    else:
        raise NotImplementedError("JPEG preprocessing currently only supports SPSA attack.")

    x_adv = torch.cat([x for x, _ in adv_testset], dim=0)
    y_adv = torch.cat([y for _, y in adv_testset], dim=0)
    loader_adv = DataLoader(TensorDataset(x_adv, y_adv), batch_size=64, shuffle=False)

    correct_adv, total_adv = 0, 0
    per_class_adv = {cls: {"correct": 0, "total": 0} for cls in class_names}

    print("[*] Evaluating JPEG-transformed model on adversarial set...")
    for x, y in tqdm(loader_adv, desc="Adversarial Eval"):
        x, y = x.to(device), y.to(device)
        x_jpeg = torch.stack([jpeg_transform(img, quality) for img in x])
        with torch.no_grad():
            preds = model(x_jpeg).argmax(dim=1)
        correct_adv += (preds == y).sum().item()
        total_adv += y.size(0)
        for i in range(len(y)):
            cls = class_names[y[i].item()]
            per_class_adv[cls]["total"] += 1
            if preds[i] == y[i]:
                per_class_adv[cls]["correct"] += 1

    acc_adv = round(correct_adv / total_adv, 4)
    per_class_acc_adv = {
        cls: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0
        for cls, v in per_class_adv.items()
    }

    # Save results
    results = {
        "defense": "jpeg_preprocessing",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "accuracy_adversarial": acc_adv,
        "per_class_accuracy_clean": per_class_acc_clean,
        "per_class_accuracy_adversarial": per_class_acc_adv,
        "params": {
            "quality": quality
        }
    }

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    json_path = f"results/evasion/{attack_type}/jpeg_preprocessing_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {json_path}")

    md_path = f"results/evasion/{attack_type}/jpeg_preprocessing_report.md"
    generate_jpeg_preprocessing_report(json_file=json_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")

