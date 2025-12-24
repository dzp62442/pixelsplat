import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def _ensure_hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 4:
        color = image[..., :3].astype(np.float32)
        alpha = image[..., 3:].astype(np.float32) / 255
        image = color * alpha + 255 * (1 - alpha)
        image = image.clip(0, 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def load_info(
    info: dict,
    dataset_prefix: str,
    data_root: Path,
) -> tuple[str, Tensor, Tensor]:
    img_path = info["data_path"].replace(dataset_prefix, str(data_root))
    c2w = torch.tensor(info["sensor2lidar_transform"], dtype=torch.float32)

    lidar2cam_rotation = np.linalg.inv(np.array(info["sensor2lidar_rotation"]))
    lidar2cam_translation = np.array(info["sensor2lidar_translation"])
    lidar2cam_translation = lidar2cam_translation @ lidar2cam_rotation.T

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = lidar2cam_rotation.T
    w2c[:3, 3] = -lidar2cam_translation

    return img_path, c2w, torch.tensor(w2c, dtype=torch.float32)


def load_conditions(
    image_paths: Sequence[str],
    resolution: tuple[int, int],
    is_input: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    images = []
    masks = []
    intrinsics = []
    h_out, w_out = resolution

    for path in image_paths:
        param_path = (
            path.replace("samples", "samples_param_small")
            .replace("sweeps", "sweeps_param_small")
            .replace(".jpg", ".json")
        )
        with open(param_path) as handle:
            param = json.load(handle)
        intrinsic = np.array(param["camera_intrinsic"], dtype=np.float32)

        image_path = (
            path.replace("samples", "samples_small")
            .replace("sweeps", "sweeps_small")
        )
        image = Image.open(image_path).convert("RGB")
        w_in, h_in = image.width, image.height
        if (h_in, w_in) != (h_out, w_out):
            scale_h = h_out / h_in
            scale_w = w_out / w_in
            intrinsic[0, :] *= scale_w
            intrinsic[1, :] *= scale_h
            image = image.resize((w_out, h_out), Image.BILINEAR)
        intrinsic[0, :] /= w_out
        intrinsic[1, :] /= h_out
        image = np.asarray(image)
        image = _ensure_hwc3(image)
        images.append(image)
        intrinsics.append(intrinsic)

        if is_input:
            mask = np.ones((h_out, w_out), dtype=np.float32)
        else:
            mask_path = (
                image_path.replace("samples_small", "samples_mask_small")
                .replace("sweeps_small", "sweeps_mask_small")
                .replace(".jpg", ".png")
            )
            mask = Image.open(mask_path).convert("L")
            if (mask.height, mask.width) != (h_out, w_out):
                mask = mask.resize((w_out, h_out), Image.BILINEAR)
            mask = np.asarray(mask, dtype=np.float32) / 255
        masks.append(mask)

    images_np = np.stack(images)
    masks_np = np.stack(masks)
    intrinsics_np = np.stack(intrinsics)

    images_tensor = (
        torch.from_numpy(images_np)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        / 255.0
    )
    masks_tensor = torch.from_numpy(masks_np > 0.5)
    intrinsics_tensor = torch.from_numpy(intrinsics_np).float()

    return images_tensor, masks_tensor, intrinsics_tensor
