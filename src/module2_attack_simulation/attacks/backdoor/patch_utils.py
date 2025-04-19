import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

def generate_patch(patch_type, size, image_tensor=None, position="bottom_right"):
    """
    Generates a patch tensor of shape (C, H, W) based on the type.
    For 'random_noise', the original image tensor must be provided.
    """
    if isinstance(size, int):
        H = W = size
    else:
        H, W = size

    if patch_type == "white_square":
        return torch.ones(3, H, W)

    elif patch_type == "checkerboard":
        pattern = np.indices((H, W)).sum(axis=0) % 2
        checker = np.stack([pattern, pattern, pattern], axis=0)
        return torch.tensor(checker, dtype=torch.float32)

    elif patch_type == "random_noise":
        if image_tensor is None:
            raise ValueError("image_tensor must be provided for 'random_noise' patch type.")

        # === Determine coordinates based on position ===
        C, full_H, full_W = image_tensor.shape
        if position == "bottom_right":
            x_start = full_W - W
            y_start = full_H - H
        elif position == "bottom_left":
            x_start = 0
            y_start = full_H - H
        elif position == "top_right":
            x_start = full_W - W
            y_start = 0
        elif position == "top_left":
            x_start = 0
            y_start = 0
        elif position == "center":
            x_start = (full_W - W) // 2
            y_start = (full_H - H) // 2
        else:
            raise ValueError(f"Unknown patch position: {position}")

        # === Extract patch region from original image and add noise ===
        patch_region = image_tensor[:, y_start:y_start + H, x_start:x_start + W].clone()
        noise = torch.randn_like(patch_region) * 0.2  # noise strength
        patch = torch.clamp(patch_region + noise, 0.0, 1.0)
        return patch

    else:
        raise ValueError(f"Unknown patch type: {patch_type}")


def apply_static_patch(image_tensor, patch_tensor, position="bottom_right",blend_alpha=None):
    """
    Applies the given patch to the image tensor at the specified position.

    Args:
        image_tensor (Tensor): shape (C, H, W)
        patch_tensor (Tensor): shape (C, h, w)
        position (str): one of ["bottom_right", "bottom_left", "top_right", "top_left", "center"]
        blend_alpha (float): blending factor for alpha blending (optional)

    Returns:
        Tensor: image with patch applied
    """
    C, H, W = image_tensor.shape
    _, h, w = patch_tensor.shape

    if position == "bottom_right":
        x_start = W - w
        y_start = H - h
    elif position == "bottom_left":
        x_start = 0
        y_start = H - h
    elif position == "top_right":
        x_start = W - w
        y_start = 0
    elif position == "top_left":
        x_start = 0
        y_start = 0
    elif position == "center":
        x_start = (W - w) // 2
        y_start = (H - h) // 2
    else:
        raise ValueError(f"Unknown patch position: {position}")

    patched_image = image_tensor.clone()

    # === Blending logic ===
    if blend_alpha is not None and 0.0 < blend_alpha < 1.0:
        region = patched_image[:, y_start:y_start + h, x_start:x_start + w]
        blended = (1 - blend_alpha) * region + blend_alpha * patch_tensor
        patched_image[:, y_start:y_start + h, x_start:x_start + w] = blended
    else:
        patched_image[:, y_start:y_start + h, x_start:x_start + w] = patch_tensor

    return patched_image


def update_poisoned_sample(poisoned_dataset, index, patched_image, target_class):
    """
    Atualiza uma imagem e label no dataset, suportando Subset ou Dataset direto.

    Args:
        poisoned_dataset: Subset ou Dataset (com .targets e .data)
        index: índice no subset ou dataset
        patched_image: tensor (C, H, W), valores [0, 1]
        target_class: nova label int
    """
    # Caso seja Subset
    if hasattr(poisoned_dataset, "dataset") and hasattr(poisoned_dataset, "indices"):
        dataset = poisoned_dataset.dataset
        real_idx = poisoned_dataset.indices[index]
    else:
        dataset = poisoned_dataset
        real_idx = index

    # Processar imagem para atualizar no dataset
    image_np = patched_image.detach().cpu()
    if image_np.shape[0] in [1, 3]:  # CHW → HWC
        image_np = image_np.permute(1, 2, 0)

    if image_np.max() <= 1.0:
        image_np = image_np * 255

    image_np = image_np.byte().numpy()

    # Atualizar imagem
    if hasattr(dataset, "data") and isinstance(dataset.data, np.ndarray):
        dataset.data[real_idx] = image_np
    elif hasattr(dataset, "data") and isinstance(dataset.data, torch.Tensor):
        dataset.data[real_idx] = torch.from_numpy(image_np)
    elif hasattr(dataset, "tensors"):
        tensors = list(dataset.tensors)
        tensors[0][real_idx] = patched_image
        tensors[1][real_idx] = target_class
        dataset.tensors = tuple(tensors)
    else:
        raise TypeError("Unsupported dataset type: cannot update poisoned sample.")

    # Atualizar label
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets[real_idx] = target_class
        elif isinstance(dataset.targets, list):
            dataset.targets[real_idx] = target_class


def initialize_trigger_and_mask(in_shape, patch_ratio):
    """
    Create a trigger T and mask M for a given input shape.
    - in_shape: (C, H, W)
    - patch_ratio: float between 0 and 1 ⇒ patch height = H*ratio, width = W*ratio
    Returns:
        T: Tensor[C, h, w], requires_grad=True
        M: Tensor[1, h, w], requires_grad=True
    """
    C, H, W = in_shape
    h = max(1, int(H * patch_ratio))
    w = max(1, int(W * patch_ratio))
    # Trigger: small random init
    T = torch.randn(C, h, w, requires_grad=True)
    # Mask: zeros => sigmoid(M)=0.5 initial
    M = torch.zeros(1, h, w, requires_grad=True)
    return T, M

def apply_trigger(x, T, M):
    """
    Apply learned trigger T and mask M to a batch of images x.
    x: Tensor[B, C, H, W]
    T: Tensor[C, h, w]
    M: Tensor[1, h, w]
    Returns: x_poisoned: Tensor[B, C, H, W]
    """
    sigM = torch.sigmoid(M)              # shape [1, h, w]
    B, C, H, W = x.shape
    _, _, h, w = (1, *T.shape)  # use T.shape for h,w
    # Create zero‐padded copies at top‐left
    padded_M = torch.zeros(B, 1, H, W, device=x.device)
    padded_T = torch.zeros(B, C, H, W, device=x.device)
    padded_M[:, :, :h, :w] = sigM
    padded_T[:, :, :h, :w] = T

    return (1 - padded_M) * x + padded_M * padded_T

def mask_l1_loss(M):
    """
    Compute L1 norm of the soft mask.
    M: Tensor[1, h, w]
    Returns: scalar Tensor = sum(|sigmoid(M)|)
    """
    return torch.abs(torch.sigmoid(M)).sum()

def total_variation_loss(T):
    """
    Compute isotropic total variation loss on trigger tensor T.
    T: Tensor[C, h, w]
    Returns: scalar Tensor = sum(|T[:, i+1, j] - T[:, i, j]| + |T[:, i, j+1] - T[:, i, j]|)
    """
    dh = torch.abs(T[:, 1:, :] - T[:, :-1, :]).sum()
    dw = torch.abs(T[:, :, 1:] - T[:, :, :-1]).sum()
    return dh + dw

def save_trigger_visualization(T, M, out_dir):
    """
    Save visualizations to out_dir:
      - mask.png   : sigmoid(M) as grayscale
      - trigger.png: patch T normalized
      - overlay.png: trigger overlaid on a mid-gray canvas
    """
    os.makedirs(out_dir, exist_ok=True)

    # Normalize T to [0,1]
    T_min, T_max = T.min(), T.max()
    T_norm = (T - T_min) / (T_max - T_min + 1e-8)

    # Mask image
    mask_img = torch.sigmoid(M)[0].cpu().numpy()
    plt.imsave(os.path.join(out_dir, "mask.png"), mask_img, cmap="gray")

    # Trigger image
    if T_norm.shape[0] == 3:
        trig_img = T_norm.permute(1, 2, 0).cpu().numpy()  # H,W,C
        trig_img = np.clip(trig_img, 0.0, 1.0)
        plt.imsave(os.path.join(out_dir, "trigger.png"), trig_img)
    else:
        trig_img = T_norm[0].cpu().numpy()
        trig_img = np.clip(trig_img, 0.0, 1.0)
        plt.imsave(os.path.join(out_dir, "trigger.png"), trig_img, cmap="gray")

    # Overlay on mid-gray canvas
    B, C, h, w = 1, T.shape[0], T.shape[1], T.shape[2]
    canvas = 0.5 * torch.ones((B, C, h, w))
    overlay = apply_trigger(canvas, T, M)[0].cpu().numpy()  # [C,H,W] or [H,W]

    # Clamp overlay
    overlay = np.clip(overlay, 0.0, 1.0)

    if C == 3:
        overlay = overlay.transpose(1, 2, 0)  # H,W,C
        plt.imsave(os.path.join(out_dir, "overlay.png"), overlay)
    else:
        overlay = overlay.mean(0)
        plt.imsave(os.path.join(out_dir, "overlay.png"), overlay, cmap="gray")




