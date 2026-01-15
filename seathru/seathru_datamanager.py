from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SeathruDataManagerConfig(VanillaDataManagerConfig):
    """Data manager with optional SeaSplat/DeepSeeColor supervision."""

    _target: Type = field(default_factory=lambda: SeathruDataManager)
    use_clean_supervision: bool = False
    use_depth_supervision: bool = False
    clean_dirname: str = "seasplat_clean"
    depth_dirname: str = "seasplat_depth"
    depth_ext: Literal["npy", "exr", "png"] = "npy"
    preload_supervision: bool = True
    strict_supervision: bool = False
    resize_to_image: bool = True


class SeathruDataManager(VanillaDataManager):
    """VanillaDataManager with optional clean/depth pixel supervision."""

    config: SeathruDataManagerConfig

    _clean_paths: List[Optional[Path]]
    _depth_paths: List[Optional[Path]]
    _clean_cache: List[Optional[torch.Tensor]]
    _depth_cache: List[Optional[torch.Tensor]]
    _mask_cache: List[Optional[torch.Tensor]]
    _depth_clip_lo: List[Optional[float]]
    _depth_clip_hi: List[Optional[float]]
    _depth_dir_example: Optional[Path]
    _clean_dir_exists: bool
    _depth_dir_exists: bool
    _clean_found_count: int
    _depth_found_count: int
    _clean_loaded_count: int
    _depth_loaded_count: int

    def setup_train(self) -> None:
        super().setup_train()
        self._setup_supervision()

    def _setup_supervision(self) -> None:
        self._clean_paths = []
        self._depth_paths = []
        self._clean_cache = []
        self._depth_cache = []
        self._mask_cache = []
        self._depth_clip_lo = []
        self._depth_clip_hi = []
        self._depth_dir_example = None
        self._clean_dir_exists = False
        self._depth_dir_exists = False
        self._clean_found_count = 0
        self._depth_found_count = 0
        self._clean_loaded_count = 0
        self._depth_loaded_count = 0

        if not (self.config.use_clean_supervision or self.config.use_depth_supervision):
            return

        image_paths = self.train_dataset.image_filenames
        for image_path in image_paths:
            parent_name = image_path.parent.name.lower()
            if parent_name.startswith("images"):
                data_root = image_path.parent.parent
            else:
                data_root = image_path.parent

            clean_dir = data_root / self.config.clean_dirname
            depth_dir = data_root / self.config.depth_dirname
            if clean_dir.is_dir():
                self._clean_dir_exists = True
            if depth_dir.is_dir():
                self._depth_dir_exists = True
                if self._depth_dir_example is None:
                    self._depth_dir_example = depth_dir

            clean_path = clean_dir / f"{image_path.stem}.png"
            depth_path = depth_dir / f"{image_path.stem}.{self.config.depth_ext}"

            if self.config.use_clean_supervision and clean_path.exists():
                self._clean_paths.append(clean_path)
                self._clean_found_count += 1
            else:
                self._clean_paths.append(None)

            if self.config.use_depth_supervision and depth_path.exists():
                self._depth_paths.append(depth_path)
                self._depth_found_count += 1
            else:
                self._depth_paths.append(None)

        if self.config.strict_supervision:
            if self.config.use_clean_supervision and not self._clean_dir_exists:
                raise RuntimeError("clean supervision directory not found")
            if self.config.use_depth_supervision and not self._depth_dir_exists:
                raise RuntimeError("depth supervision directory not found")

        num_images = len(image_paths)
        self._clean_cache = [None] * num_images
        self._depth_cache = [None] * num_images
        self._mask_cache = [None] * num_images
        self._depth_clip_lo = [None] * num_images
        self._depth_clip_hi = [None] * num_images

        if self.config.preload_supervision:
            for idx in range(num_images):
                if self.config.use_clean_supervision:
                    clean_tensor = self._get_clean_for_cam(idx)
                    if clean_tensor is not None:
                        self._clean_loaded_count += 1
                if self.config.use_depth_supervision:
                    depth_tensor = self._get_depth_for_cam(idx)
                    if depth_tensor is not None:
                        self._depth_loaded_count += 1

        if self.config.use_clean_supervision:
            missing = num_images - self._clean_found_count
            status = "found" if self._clean_dir_exists else "missing"
            CONSOLE.print(
                f"[SeathruDataManager] clean dir {status}, "
                f"loaded {self._clean_loaded_count}, missing {missing}"
            )
        if self.config.use_depth_supervision:
            missing = num_images - self._depth_found_count
            status = "found" if self._depth_dir_exists else "missing"
            depth_path = str(self._depth_dir_example) if self._depth_dir_example is not None else "None"
            example = ""
            for lo, hi in zip(self._depth_clip_lo, self._depth_clip_hi):
                if lo is not None and hi is not None:
                    example = f", lo={lo:.6f}, hi={hi:.6f}"
                    break
            CONSOLE.print(
                f"[SeathruDataManager] depth dir {status}, path {depth_path}, "
                f"loaded {self._depth_loaded_count}, missing {missing}{example}"
            )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        indices = batch.get("indices")
        if indices is None:
            return ray_bundle, batch

        if self.config.use_clean_supervision and self._clean_found_count > 0:
            clean = self._gather_clean(indices)
            if clean is not None:
                batch["clean_image"] = clean

        if self.config.use_depth_supervision and self._depth_found_count > 0:
            depth, depth_mask = self._gather_depth(indices)
            if depth is not None and depth_mask is not None:
                batch["depth"] = depth
                batch["depth_mask"] = depth_mask

        return ray_bundle, batch

    def _get_image_hw(self, cam_idx: int) -> Optional[Tuple[int, int]]:
        cameras = self.train_dataset.cameras
        height = cameras.height
        width = cameras.width
        try:
            if torch.is_tensor(height):
                h = int(height[cam_idx].item())
                w = int(width[cam_idx].item())
            else:
                h = int(height)
                w = int(width)
            return h, w
        except Exception:
            image_path = self.train_dataset.image_filenames[cam_idx]
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
                return h, w
            except Exception:
                return None

    def _get_clean_for_cam(self, cam_idx: int) -> Optional[torch.Tensor]:
        if cam_idx < 0 or cam_idx >= len(self._clean_cache):
            return None
        cached = self._clean_cache[cam_idx]
        if cached is not None:
            return cached
        path = self._clean_paths[cam_idx]
        if path is None:
            if self.config.strict_supervision:
                raise RuntimeError(f"Missing clean supervision for cam {cam_idx}")
            return None
        target_hw = self._get_image_hw(cam_idx) if self.config.resize_to_image else None
        clean = self._load_clean_tensor(path, target_hw)
        if clean is None:
            if self.config.strict_supervision:
                raise RuntimeError(f"Failed to load clean supervision for cam {cam_idx}")
            return None
        self._clean_cache[cam_idx] = clean
        return clean

    def _get_depth_for_cam(self, cam_idx: int) -> Optional[torch.Tensor]:
        if cam_idx < 0 or cam_idx >= len(self._depth_cache):
            return None
        cached = self._depth_cache[cam_idx]
        if cached is not None:
            return cached
        path = self._depth_paths[cam_idx]
        if path is None:
            if self.config.strict_supervision:
                raise RuntimeError(f"Missing depth supervision for cam {cam_idx}")
            return None
        target_hw = self._get_image_hw(cam_idx) if self.config.resize_to_image else None
        depth, mask, lo, hi = self._load_depth_tensor(path, target_hw)
        if depth is None or mask is None:
            if self.config.strict_supervision:
                raise RuntimeError(f"Failed to load depth supervision for cam {cam_idx}")
            return None
        self._depth_cache[cam_idx] = depth
        self._mask_cache[cam_idx] = mask
        if lo is not None and hi is not None:
            self._depth_clip_lo[cam_idx] = lo
            self._depth_clip_hi[cam_idx] = hi
        return depth

    def _load_clean_tensor(self, path: Path, target_hw: Optional[Tuple[int, int]]) -> Optional[torch.Tensor]:
        try:
            image = Image.open(path).convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            arr = arr[:, :, :3]
            clean = torch.from_numpy(arr)
            if target_hw is not None and (clean.shape[0] != target_hw[0] or clean.shape[1] != target_hw[1]):
                clean = clean.permute(2, 0, 1).unsqueeze(0)
                clean = F.interpolate(clean, size=target_hw, mode="bilinear", align_corners=False)
                clean = clean.squeeze(0).permute(1, 2, 0).contiguous()
            return clean
        except Exception:
            return None

    def _load_depth_tensor(
        self, path: Path, target_hw: Optional[Tuple[int, int]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[float]]:
        lo = None
        hi = None
        try:
            if self.config.depth_ext == "npy":
                depth = np.load(path)
            else:
                depth = np.asarray(Image.open(path))
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth.ndim != 2:
                return None, None, None, None
            depth = depth.astype(np.float32)
            base_valid = np.isfinite(depth) & (depth > 0)
            valid_count = int(base_valid.sum())
            if valid_count >= 1000:
                try:
                    lo_val, hi_val = np.quantile(depth[base_valid], [0.001, 0.999])
                    if np.isfinite(lo_val) and np.isfinite(hi_val) and hi_val > lo_val:
                        ratio = lo_val / hi_val
                        if np.isfinite(ratio):
                            lo = float(lo_val)
                            hi = float(hi_val)
                except Exception:
                    lo = None
                    hi = None
            if lo is not None and hi is not None:
                depth = np.clip(depth, lo, hi)
                mask = base_valid & (depth >= lo) & (depth <= hi)
            else:
                mask = base_valid
            depth_t = torch.from_numpy(depth)
            mask_t = torch.from_numpy(mask)
            if target_hw is not None and (depth_t.shape[0] != target_hw[0] or depth_t.shape[1] != target_hw[1]):
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                mask_f = mask_t.to(torch.float32).unsqueeze(0).unsqueeze(0)
                depth_t = F.interpolate(depth_t, size=target_hw, mode="nearest")
                mask_f = F.interpolate(mask_f, size=target_hw, mode="nearest")
                depth_t = depth_t.squeeze(0).squeeze(0).contiguous()
                mask_t = mask_f.squeeze(0).squeeze(0) > 0.5
            return depth_t, mask_t, lo, hi
        except Exception:
            return None, None, None, None

    def _gather_clean(self, indices: torch.Tensor) -> Optional[torch.Tensor]:
        indices_cpu = indices.detach().to("cpu")
        cam_idx = indices_cpu[:, 0].long()
        y = indices_cpu[:, 1].long()
        x = indices_cpu[:, 2].long()
        if cam_idx.numel() == 0:
            return None
        if int(cam_idx.min()) < 0 or int(cam_idx.max()) >= len(self._clean_cache):
            if self.config.strict_supervision:
                raise RuntimeError("clean cam_idx out of range")
            return None

        output = None
        for cam in torch.unique(cam_idx):
            cam_i = int(cam)
            clean = self._get_clean_for_cam(cam_i)
            if clean is None:
                return None
            height, width = clean.shape[0], clean.shape[1]
            mask = cam_idx == cam_i
            if bool((y[mask] >= height).any()) or bool((x[mask] >= width).any()):
                if self.config.strict_supervision:
                    raise RuntimeError("clean indices out of bounds")
                return None
            if output is None:
                output = torch.empty((cam_idx.shape[0], 3), dtype=clean.dtype)
            output[mask] = clean[y[mask], x[mask]]
        if output is not None and indices.device != output.device:
            output = output.to(indices.device)
        return output

    def _gather_depth(self, indices: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        indices_cpu = indices.detach().to("cpu")
        cam_idx = indices_cpu[:, 0].long()
        y = indices_cpu[:, 1].long()
        x = indices_cpu[:, 2].long()
        if cam_idx.numel() == 0:
            return None, None
        if int(cam_idx.min()) < 0 or int(cam_idx.max()) >= len(self._depth_cache):
            if self.config.strict_supervision:
                raise RuntimeError("depth cam_idx out of range")
            return None, None

        depth_out = None
        mask_out = None
        for cam in torch.unique(cam_idx):
            cam_i = int(cam)
            depth = self._get_depth_for_cam(cam_i)
            mask = self._mask_cache[cam_i]
            if depth is None or mask is None:
                return None, None
            height, width = depth.shape[0], depth.shape[1]
            sel = cam_idx == cam_i
            if bool((y[sel] >= height).any()) or bool((x[sel] >= width).any()):
                if self.config.strict_supervision:
                    raise RuntimeError("depth indices out of bounds")
                return None, None
            if depth_out is None:
                depth_out = torch.empty((cam_idx.shape[0], 1), dtype=depth.dtype)
                mask_out = torch.empty((cam_idx.shape[0], 1), dtype=torch.bool)
            depth_out[sel, 0] = depth[y[sel], x[sel]]
            mask_out[sel, 0] = mask[y[sel], x[sel]]

        if depth_out is not None and indices.device != depth_out.device:
            depth_out = depth_out.to(indices.device)
            mask_out = mask_out.to(indices.device)
        return depth_out, mask_out
