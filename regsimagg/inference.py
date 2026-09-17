"""Inference pipeline for generating NIfTI segmentation masks for challenge evaluation.

Corresponds to `model_outputs_to_disc` in the FeTS 2022 Challenge code.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import nibabel as nib
import pandas as pd
import torch

from .model import get_model, Small3DUNet, convert_subregion_logits_to_labels

logger = logging.getLogger(__name__)

try:
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


def model_outputs_to_disc(
    data_path: str | Path,
    output_path: str | Path,
    native_model_path: str | Path,
    validation_csv: Optional[str] = None,
    outputtag: str = "",
    device: str = "cpu",
    model_type: Optional[str] = None,
    roi_size: tuple = (64, 64, 64),
    sw_batch_size: int = 1,
    limit: Optional[int] = None,
) -> Path:
    """Run inference over validation dataset and save NIfTI predictions to disk.

    Matches the FeTS 2022 Challenge output generation format for leaderboard submission.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_path)

    # Load model weights
    state = torch.load(native_model_path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if model_type is None:
        first_keys = list(state.keys())
        if any("enc1" in k for k in first_keys):
            base_ch = int(state["enc1.0.weight"].shape[0])
            model = Small3DUNet(base=base_ch).to(dev)
        else:
            model = get_model(model_type="monai_unet").to(dev)
    else:
        model = get_model(model_type=model_type).to(dev)
    model.load_state_dict(state)
    model.eval()

    # Find cases to infer
    subjects = []
    if validation_csv:
        csv_file = data_dir / validation_csv if not Path(validation_csv).is_absolute() else Path(validation_csv)
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            first_col = df.columns[0]
            subjects = [str(val).strip() for val in df[first_col]]

    if not subjects:
        # Check subdirectories first (e.g. FeTS2022_XXXXX)
        subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and (d.name.startswith("FeTS") or list(d.glob("*.nii*")))])
        if subdirs:
            subjects = subdirs
        else:
            # Fallback to flat files
            nii_files = list(data_dir.glob("*.nii*"))
            subjects = sorted(list(set(f.name.replace(".nii.gz", "").replace(".nii", "").rsplit("_", 1)[0] for f in nii_files)))

    if limit is not None and limit > 0:
        subjects = subjects[:limit]

    logger.info("Found %d subjects for inference.", len(subjects))

    with torch.no_grad():
        for sid in subjects:
            s_dir = data_dir / sid if (data_dir / sid).is_dir() else data_dir
            t1ce = list(s_dir.glob(f"*{sid}*t1ce*.nii*")) or list(s_dir.glob(f"*t1ce*.nii*"))
            t1 = [p for p in (list(s_dir.glob(f"*{sid}*t1*.nii*")) or list(s_dir.glob(f"*t1*.nii*"))) if "t1ce" not in p.name.lower()]
            t2 = list(s_dir.glob(f"*{sid}*t2*.nii*")) or list(s_dir.glob(f"*t2*.nii*"))
            flair = list(s_dir.glob(f"*{sid}*flair*.nii*")) or list(s_dir.glob(f"*flair*.nii*"))

            if t1 and t1ce and t2 and flair:
                four_mods = [t1[0], t1ce[0], t2[0], flair[0]]
                ref_nii = nib.load(str(t1[0]))
                affine = ref_nii.affine
                header = ref_nii.header
                img_data = np.stack([nib.load(str(p)).get_fdata(dtype=np.float32) for p in four_mods], axis=0)
            else:
                matches = sorted(data_dir.glob(f"**/{sid}*.nii*"))
                if not matches:
                    continue
                ref_nii = nib.load(str(matches[0]))
                affine = ref_nii.affine
                header = ref_nii.header
                img_data = ref_nii.get_fdata(dtype=np.float32)
                if img_data.ndim == 4 and img_data.shape[-1] == 4:
                    img_data = np.moveaxis(img_data, -1, 0)
                elif img_data.ndim == 3:
                    img_data = np.stack([img_data] * 4, axis=0)

            # Normalize non-zero voxels
            for c in range(img_data.shape[0]):
                m = img_data[c] > 0
                if m.any():
                    mean, std = img_data[c][m].mean(), img_data[c][m].std()
                    img_data[c][m] = (img_data[c][m] - mean) / (std + 1e-6)

            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(dev)

            effective_overlap = 0.125 if dev.type == "cpu" else 0.25

            if HAS_MONAI:
                logits = sliding_window_inference(
                    inputs=inp,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=effective_overlap,
                )
            else:
                logits = model(inp)

            # Convert subregion logits [WT, TC, ET] back to BraTS labels {0, 1, 2, 4}
            pred_labels = convert_subregion_logits_to_labels(logits)[0].cpu().numpy().astype(np.int16)

            # Save prediction NIfTI file
            out_filename = f"{sid}{outputtag}.nii.gz"
            pred_nii = nib.Nifti1Image(pred_labels, affine, header)
            nib.save(pred_nii, str(out_dir / out_filename))

    logger.info("Inference completed. Outputs saved to %s", out_dir)
    return out_dir
