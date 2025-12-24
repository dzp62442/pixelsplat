import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .utils_omniscene import load_conditions, load_info
from .view_sampler import ViewSampler


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    make_baseline_1: bool
    augment: bool
    baseline_epsilon: float
    max_fov: float
    near: float
    far: float
    baseline_scale_bounds: bool
    shuffle_val: bool
    test_len: int
    train_times_per_scene: int
    skip_bad_shape: bool
    highres: bool


class DatasetOmniScene(Dataset):
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    dataset_prefix = "/datasets/nuScenes"
    data_version = "interp_12Hz_trainval"

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        del view_sampler
        self.cfg = cfg
        self.stage = stage
        self.data_root = Path(cfg.roots[0])
        self.resolution = tuple(cfg.image_shape)
        self.bin_tokens = self._load_bin_tokens(stage)

    def _load_bin_tokens(self, stage: Stage) -> list[str]:
        base = self.data_root / self.data_version

        def load_file(name: str) -> list[str]:
            with open(base / name) as handle:
                return json.load(handle)["bins"]

        if stage == "train":
            bins = load_file("bins_train_3.2m.json")
            bins = bins * max(1, self.cfg.train_times_per_scene)
        else:
            bins = load_file("bins_val_3.2m.json")
            if stage == "val":
                bins = bins[:30_000:3_000][:10]
            elif stage == "test":
                bins = bins[0::14][:2048]

        if stage == "test" and self.cfg.test_len > 0:
            bins = bins[: self.cfg.test_len]

        if self.cfg.overfit_to_scene is not None:
            bins = [self.cfg.overfit_to_scene]

        return bins

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        info_path = (
            self.data_root / self.data_version / "bin_infos_3.2m" / f"{bin_token}.pkl"
        )
        with open(info_path, "rb") as handle:
            bin_info = pickle.load(handle)

        sensor_info = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        context_image_paths: list[str] = []
        context_extrinsics: list[Tensor] = []
        for camera in self.camera_types:
            img_path, c2w, _ = load_info(
                sensor_info[camera],
                self.dataset_prefix,
                self.data_root,
            )
            context_image_paths.append(img_path)
            context_extrinsics.append(c2w)
        context_extrinsics_tensor = torch.stack(context_extrinsics)

        (
            context_images_tensor,
            context_masks_tensor,
            context_intrinsics_tensor,
        ) = load_conditions(
            context_image_paths,
            self.resolution,
            is_input=True,
        )
        context_masks_tensor = context_masks_tensor.bool()

        render_image_paths: list[str] = []
        render_extrinsics: list[Tensor] = []
        frame_count = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_count < 3:
            raise ValueError(f"bin {bin_token} has insufficient frames.")

        for camera in self.camera_types:
            for offset in (1, 2):
                info = bin_info["sensor_info"][camera][offset]
                img_path, c2w, _ = load_info(
                    info,
                    self.dataset_prefix,
                    self.data_root,
                )
                render_image_paths.append(img_path)
                render_extrinsics.append(c2w)
        render_extrinsics_tensor = torch.stack(render_extrinsics)

        (
            render_images_tensor,
            render_masks_tensor,
            render_intrinsics_tensor,
        ) = load_conditions(
            render_image_paths,
            self.resolution,
            is_input=False,
        )
        render_masks_tensor = render_masks_tensor.bool()

        target_images = torch.cat([render_images_tensor, context_images_tensor], dim=0)
        target_masks = torch.cat([render_masks_tensor, context_masks_tensor], dim=0)
        target_extrinsics = torch.cat(
            [render_extrinsics_tensor, context_extrinsics_tensor], dim=0
        )
        target_intrinsics = torch.cat(
            [render_intrinsics_tensor, context_intrinsics_tensor], dim=0
        )

        near = torch.tensor(self.cfg.near, dtype=torch.float32)
        far = torch.tensor(self.cfg.far, dtype=torch.float32)

        context = {
            "extrinsics": context_extrinsics_tensor,
            "intrinsics": context_intrinsics_tensor,
            "image": context_images_tensor,
            "near": repeat(near, " -> v", v=context_images_tensor.shape[0]),
            "far": repeat(far, " -> v", v=context_images_tensor.shape[0]),
            "index": torch.arange(context_images_tensor.shape[0]),
        }
        target = {
            "extrinsics": target_extrinsics,
            "intrinsics": target_intrinsics,
            "image": target_images,
            "near": repeat(near, " -> v", v=target_images.shape[0]),
            "far": repeat(far, " -> v", v=target_images.shape[0]),
            "index": torch.arange(target_images.shape[0]),
            "masks": target_masks,
        }
        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
