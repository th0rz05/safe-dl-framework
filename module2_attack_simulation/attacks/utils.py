import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import shutil

def train_model(model, trainset, valset=None, epochs=3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size) if valset else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train loss = {running_loss:.4f}")

        if val_loader:
            acc = evaluate_model(model, valset)
            print(f"         Validation accuracy = {acc:.4f}")

def evaluate_model(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=64)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def get_class_labels(dataset):
    
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        targets = dataset.dataset.targets
    elif hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        raise ValueError("Dataset does not expose `.targets` attribute.")

    unique_labels = sorted(set(int(label) for label in targets))
    return unique_labels



def save_flip_examples(dataset, flip_log, output_dir="results/flipped_examples", num_examples=5):
    # Apaga o diretório anterior, se existir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(flip_log[:num_examples]):
        idx = entry["index"]
        original = entry["original_label"]
        new = entry["new_label"]
        img, _ = dataset[idx]

        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"{original} -> {new}")
        plt.axis("off")
        filename = f"sample_{i}_{original}_to_{new}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
    

def save_flip_examples(dataset, flip_log, num_examples=5, output_dir="results/flipped_examples"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for example in flip_log:
        try:
            idx = example["index"]
            original_label = example["original_label"]
            new_label = example["new_label"]
            img, _ = dataset[idx]  # img: tensor [C, H, W]

            # Convert image to NumPy array and rearrange if needed
            img = img.cpu()

            if img.shape[0] in [3, 4]:  # RGB or RGBA
                img = img.permute(1, 2, 0)  # [H, W, C]

            img = img.squeeze()

            # Save the image with appropriate color mapping
            plt.figure(figsize=(3, 3))
            plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
            plt.axis("off")
            plt.title(f"{original_label} → {new_label}")
            filename = os.path.join(output_dir, f"flip_{idx}_{original_label}_to_{new_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            
            
            saved += 1
            if saved >= num_examples:
                break

        except Exception as e:
            print(f"[!] Failed to save flip example for index {example.get('index', '?')}: {e}")
