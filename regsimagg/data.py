"""Data loading and preprocessing pipelines for FeTS / BraTS 3D MRI volumes.

Supports:
1. 4-channel multi-modal NIfTI files (T1, T1ce, T2, FLAIR).
2. Separate per-modality NIfTI files.
3. Subregion target conversion (WT, TC, ET).
4. MONAI-based intensity normalization and spatial cropping/padding.
5. Synthetic volume generator for automated testing and quick reproduction.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple
import logging
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from .model import convert_labels_to_subregions

logger = logging.getLogger(__name__)


class FeTSNiftiDataset(Dataset):
    """Dataset for multi-modal brain tumor MRI scans."""

    def __init__(
        self,
        client_dir: str | Path,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        convert_to_subregions: bool = True,
    ):
        self.client_dir = Path(client_dir)
        self.patch_size = patch_size
        self.convert_to_subregions = convert_to_subregions

        self.images_dir = self.client_dir / "imagesTr"
        self.labels_dir = self.client_dir / "labelsTr"

        self.subjects = []
        if self.labels_dir.exists():
            self.subjects = sorted(self.labels_dir.glob("*.nii*"))
        elif self.images_dir.exists():
            self.subjects = sorted(self.images_dir.glob("*.nii*"))

    def __len__(self) -> int:
        return len(self.subjects)

    def _resolve_image_channels(self, label_path: Path) -> np.ndarray:
        stem = label_path.name.replace(".nii.gz", "").replace(".nii", "")

        # Case 1: Four separate modality files (e.g. stem_0000.nii.gz ... stem_0003.nii.gz)
        four_numbered = [self.images_dir / f"{stem}_{i:04d}.nii.gz" for i in range(4)]
        if all(p.exists() for p in four_numbered):
            return np.stack([nib.load(str(p)).get_fdata(dtype=np.float32) for p in four_numbered], axis=0)

        # Case 2: Named modalities (_t1, _t1ce, _t2, _flair)
        named_modalities = [
            self.images_dir / f"{stem}_{mod}.nii.gz"
            for mod in ["t1", "t1ce", "t2", "flair"]
        ]
        if all(p.exists() for p in named_modalities):
            return np.stack([nib.load(str(p)).get_fdata(dtype=np.float32) for p in named_modalities], axis=0)

        # Case 3: Single 4D file containing all 4 channels
        candidates = sorted(self.images_dir.glob(f"{stem}*.nii*"))
        if not candidates:
            # Check direct match
            candidates = sorted(self.client_dir.glob(f"**/{stem}*.nii*"))

        if not candidates:
            raise FileNotFoundError(f"Could not find matching image for {stem} in {self.images_dir}")

        img_arr = nib.load(str(candidates[0])).get_fdata(dtype=np.float32)
        if img_arr.ndim == 4 and img_arr.shape[-1] == 4:
            img_arr = np.moveaxis(img_arr, -1, 0)
        elif img_arr.ndim == 4 and img_arr.shape[0] == 4:
            pass
        elif img_arr.ndim == 3:
            # Single channel duplicated 4 times (for smoke tests)
            img_arr = np.stack([img_arr] * 4, axis=0)
        else:
            raise ValueError(f"Expected 4 channels for MRI volume, got shape {img_arr.shape}")

        return img_arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label_path = self.subjects[idx]
        x = self._resolve_image_channels(label_path)

        if label_path.exists() and "labels" in str(label_path.parent):
            y = nib.load(str(label_path)).get_fdata().astype(np.int64)
        else:
            y = np.zeros(x.shape[1:], dtype=np.int64)

        # Non-zero intensity normalization per channel
        for c in range(x.shape[0]):
            channel = x[c]
            mask = channel > 0
            if mask.any():
                mean = channel[mask].mean()
                std = channel[mask].std()
                channel[mask] = (channel[mask] - mean) / (std + 1e-6)
            x[c] = channel

        # Center crop / pad to target patch size
        x = self._center_crop_pad(x, self.patch_size)
        y = self._center_crop_pad(y[None], self.patch_size)[0]

        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()

        if self.convert_to_subregions:
            # Convert {0, 1, 2, 4} to 3 binary subregion channels [WT, TC, ET]
            y_target = convert_labels_to_subregions(y_tensor.unsqueeze(0))[0]
        else:
            y_target = y_tensor

        return x_tensor, y_target

    @staticmethod
    def _center_crop_pad(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
        out = np.zeros((arr.shape[0], *target), dtype=arr.dtype)
        src_slices = []
        dst_slices = []
        for dim, t in zip(arr.shape[1:], target):
            src_len = min(dim, t)
            start = max((dim - src_len) // 2, 0)
            dstart = max((t - src_len) // 2, 0)
            src_slices.append(slice(start, start + src_len))
            dst_slices.append(slice(dstart, dstart + src_len))
        out[(slice(None), *dst_slices)] = arr[(slice(None), *src_slices)]
        return out


def crop_patch_pos_neg(
    img: np.ndarray,
    lbl: np.ndarray,
    patch_size: Tuple[int, int, int],
    is_train: bool = True,
    pos_ratio: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops a 3D patch with foreground tumor oversampling during training."""
    dims = img.shape[1:]

    if is_train:
        tumor_indices = np.argwhere(lbl > 0)
        if len(tumor_indices) > 0 and np.random.rand() < pos_ratio:
            center = tumor_indices[np.random.randint(len(tumor_indices))]
        else:
            brain_indices = np.argwhere(img[0] > 0)
            if len(brain_indices) > 0 and np.random.rand() < 0.9:
                center = brain_indices[np.random.randint(len(brain_indices))]
            else:
                center = [d // 2 for d in dims]
    else:
        tumor_indices = np.argwhere(lbl > 0)
        if len(tumor_indices) > 0:
            center = np.mean(tumor_indices, axis=0).astype(int)
        else:
            brain_indices = np.argwhere(img[0] > 0)
            if len(brain_indices) > 0:
                center = np.mean(brain_indices, axis=0).astype(int)
            else:
                center = [d // 2 for d in dims]

    starts = []
    ends = []
    for d, c, p in zip(dims, center, patch_size):
        s = max(0, min(int(c - p // 2), max(0, d - p)))
        e = min(d, s + p)
        starts.append(s)
        ends.append(e)

    s_x, s_y, s_z = starts
    e_x, e_y, e_z = ends
    cropped_img = img[:, s_x:e_x, s_y:e_y, s_z:e_z]
    cropped_lbl = lbl[s_x:e_x, s_y:e_y, s_z:e_z]

    if cropped_img.shape[1:] != patch_size:
        cropped_img = FeTSNiftiDataset._center_crop_pad(cropped_img, patch_size)
        cropped_lbl = FeTSNiftiDataset._center_crop_pad(cropped_lbl[None], patch_size)[0]

    return cropped_img, cropped_lbl


class FeTSChallengeDataset(Dataset):
    """Direct in-place dataset for the official MICCAI FeTS 2022 dataset structure.

    Reads from `MICCAI_FeTS2022_TrainingData` or `MICCAI_FeTS2022_ValidationData`
    using subject directories `FeTS2022_XXXXX` and partition CSV (`partitioning_1.csv` or `partitioning_2.csv`).
    Includes in-memory volume caching and tumor-centered patch extraction.
    """

    def __init__(
        self,
        data_root: str | Path,
        partition_csv: Optional[str | Path] = None,
        partition_id: Optional[int | str] = None,
        subject_ids: Optional[Sequence[str]] = None,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        convert_to_subregions: bool = True,
        is_train: bool = True,
        cache_capacity: int = 12,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.convert_to_subregions = convert_to_subregions
        self.is_train = is_train
        self.cache_capacity = cache_capacity
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if subject_ids is not None:
            self.subject_ids = [str(s).strip() for s in subject_ids]
        elif partition_csv is not None and partition_id is not None:
            import pandas as pd
            df = pd.read_csv(partition_csv)
            subj_col = [c for c in df.columns if "subject" in c.lower()][0]
            part_col = [c for c in df.columns if "partition" in c.lower()][0]
            sub_df = df[df[part_col].astype(str) == str(partition_id)]
            self.subject_ids = [str(s).strip() for s in sub_df[subj_col]]
        else:
            self.subject_ids = sorted([
                d.name for d in self.data_root.iterdir()
                if d.is_dir() and (d.name.startswith("FeTS") or list(d.glob("*.nii*")))
            ])

        self.cases = []
        for sid in self.subject_ids:
            s_dir = self.data_root / sid if (self.data_root / sid).is_dir() else self.data_root
            t1ce = list(s_dir.glob(f"*{sid}*t1ce*.nii*")) or list(s_dir.glob(f"*t1ce*.nii*"))
            t1 = [p for p in (list(s_dir.glob(f"*{sid}*t1*.nii*")) or list(s_dir.glob(f"*t1*.nii*"))) if "t1ce" not in p.name.lower()]
            t2 = list(s_dir.glob(f"*{sid}*t2*.nii*")) or list(s_dir.glob(f"*t2*.nii*"))
            flair = list(s_dir.glob(f"*{sid}*flair*.nii*")) or list(s_dir.glob(f"*flair*.nii*"))
            seg = list(s_dir.glob(f"*{sid}*seg*.nii*")) or list(s_dir.glob(f"*seg*.nii*"))

            if t1 and t1ce and t2 and flair:
                self.cases.append({
                    "id": sid,
                    "modalities": [t1[0], t1ce[0], t2[0], flair[0]],
                    "seg": seg[0] if seg else None,
                })

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.cases[idx]
        sid = item["id"]

        if sid in self._cache:
            img, lbl = self._cache[sid]
        else:
            arrs = [nib.load(str(p)).get_fdata(dtype=np.float32) for p in item["modalities"]]
            img = np.stack(arrs, axis=0)

            for c in range(4):
                ch = img[c]
                mask = ch > 0
                if mask.any():
                    mean, std = ch[mask].mean(), ch[mask].std()
                    ch[mask] = (ch[mask] - mean) / (std + 1e-6)
                img[c] = ch

            if item["seg"] is not None:
                lbl = nib.load(str(item["seg"])).get_fdata().astype(np.int64)
            else:
                lbl = np.zeros(img.shape[1:], dtype=np.int64)

            if len(self._cache) < self.cache_capacity:
                self._cache[sid] = (img, lbl)

        cropped_img, cropped_lbl = crop_patch_pos_neg(
            img, lbl, self.patch_size, is_train=self.is_train
        )

        x_tensor = torch.from_numpy(cropped_img).float()
        y_tensor = torch.from_numpy(cropped_lbl).long()

        if self.convert_to_subregions and item["seg"] is not None:
            y_target = convert_labels_to_subregions(y_tensor.unsqueeze(0))[0]
        else:
            y_target = y_tensor

        return x_tensor, y_target


def make_loader(
    client_dir: Optional[str | Path] = None,
    data_root: Optional[str | Path] = None,
    partition_csv: Optional[str | Path] = None,
    partition_id: Optional[int | str] = None,
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    num_workers: int = 0,
    shuffle: bool = True,
    is_train: bool = True,
) -> DataLoader:
    """Create a DataLoader supporting both client directories and official FeTS challenge layout."""
    if data_root is not None and partition_csv is not None and partition_id is not None:
        dataset = FeTSChallengeDataset(
            data_root=data_root,
            partition_csv=partition_csv,
            partition_id=partition_id,
            patch_size=patch_size,
            is_train=is_train,
        )
    elif client_dir is not None:
        c_path = Path(client_dir)
        if (c_path / "imagesTr").exists() or (c_path / "labelsTr").exists():
            dataset = FeTSNiftiDataset(client_dir=c_path, patch_size=patch_size)
        else:
            dataset = FeTSChallengeDataset(data_root=c_path, patch_size=patch_size, is_train=is_train)
    elif data_root is not None:
        dataset = FeTSChallengeDataset(data_root=data_root, patch_size=patch_size, is_train=is_train)
    else:
        raise ValueError("Must provide either client_dir or (data_root, partition_csv, partition_id)")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Synthetic NIfTI Data Generator (for reproduction smoke tests & CI)
# ---------------------------------------------------------------------------

def generate_synthetic_fets_data(
    root_dir: str | Path,
    num_clients: int = 5,
    samples_per_client: int = 2,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    seed: int = 42,
) -> Path:
    """Generate realistic synthetic 4-channel NIfTI volumes for local simulation."""
    rng = np.random.default_rng(seed)
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    for client_id in range(num_clients):
        client_dir = root / f"client_{client_id}"
        img_dir = client_dir / "imagesTr"
        lbl_dir = client_dir / "labelsTr"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in range(samples_per_client):
            subject_id = f"FeTS22_syn_{client_id:03d}_{s:03d}"

            # Create 4 structural channels: T1, T1ce, T2, FLAIR
            channels = []
            for _ in range(4):
                vol = rng.normal(loc=100.0, scale=25.0, size=volume_shape).astype(np.float32)
                vol = np.clip(vol, 0, None)
                channels.append(vol)
            img_4d = np.stack(channels, axis=-1)  # shape (D, H, W, 4)

            # Create synthetic ground truth labels {0, 1, 2, 4}
            lbl = np.zeros(volume_shape, dtype=np.int16)
            center = [dim // 2 for dim in volume_shape]
            radius = min(volume_shape) // 6

            # Peritumoral edema (ED = 2)
            z, y, x = np.ogrid[: volume_shape[0], : volume_shape[1], : volume_shape[2]]
            dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
            lbl[dist <= radius * 1.5] = 2

            # Necrotic tumor core (NCR = 1)
            lbl[dist <= radius] = 1

            # Enhancing tumor (ET = 4)
            lbl[(dist <= radius) & (dist >= radius * 0.5)] = 4

            # Save as NIfTI
            affine = np.eye(4)
            img_nii = nib.Nifti1Image(img_4d, affine)
            lbl_nii = nib.Nifti1Image(lbl, affine)

            nib.save(img_nii, str(img_dir / f"{subject_id}.nii.gz"))
            nib.save(lbl_nii, str(lbl_dir / f"{subject_id}.nii.gz"))

    logger.info("Generated synthetic dataset at %s for %d clients.", root, num_clients)
    return root
