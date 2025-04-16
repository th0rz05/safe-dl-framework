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


def apply_static_patch(image_tensor, patch_tensor, position="bottom_right"):
    """
    Applies the given patch to the image tensor at the specified position.

    Args:
        image_tensor (Tensor): shape (C, H, W)
        patch_tensor (Tensor): shape (C, h, w)
        position (str): one of ["bottom_right", "bottom_left", "top_right", "top_left", "center"]

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

    # Apply patch
    image_tensor[:, y_start:y_start + h, x_start:x_start + w] = patch_tensor
    return image_tensor


def update_poisoned_sample(poisoned_dataset, index, patched_image, target_class):
    """
    Atualiza uma imagem e label no dataset, independentemente se é built-in ou custom.

    Args:
        poisoned_dataset: Subset (com .indices e .dataset)
        index: índice no subset (não no dataset base)
        patched_image: tensor (C, H, W), valores [0, 1]
        target_class: nova label int
    """
    dataset = poisoned_dataset.dataset
    real_idx = poisoned_dataset.indices[index]

    # 1. Processar a imagem
    image_np = patched_image.detach().cpu()
    if image_np.shape[0] in [1, 3]:  # CHW → HWC
        image_np = image_np.permute(1, 2, 0)

    if image_np.max() <= 1.0:
        image_np = image_np * 255

    image_np = image_np.byte().numpy()

    # 2. Atualizar imagem
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

    # 3. Atualizar label
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets[real_idx] = target_class
        elif isinstance(dataset.targets, list):
            dataset.targets[real_idx] = target_class


