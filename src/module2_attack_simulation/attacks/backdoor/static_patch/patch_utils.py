import torch
import numpy as np

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



