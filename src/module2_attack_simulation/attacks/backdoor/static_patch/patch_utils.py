import torch
import numpy as np

def generate_patch(patch_type, size):
    """
    Generates a patch tensor of shape (C, H, W) based on the type.
    """
    if isinstance(size, int):
        H = W = size
    else:
        H, W = size

    patch = None
    if patch_type == "white_square":
        patch = torch.ones(3, H, W)

    elif patch_type == "checkerboard":
        pattern = np.indices((H, W)).sum(axis=0) % 2
        checker = np.stack([pattern, pattern, pattern], axis=0)
        patch = torch.tensor(checker, dtype=torch.float32)

    elif patch_type == "random_noise":
        patch = torch.rand(3, H, W)

    else:
        raise ValueError(f"Unknown patch type: {patch_type}")

    return patch

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

