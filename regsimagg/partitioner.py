"""FeTS challenge data partitioning utilities.

Parses official FeTS CSV partition files (e.g. partitioning_1.csv, partitioning_2.csv)
and sets up client datasets for federated training.
"""

from __future__ import annotations
from pathlib import Path
import os
import shutil
from typing import Dict, List, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {col.lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    for c in candidates:
        key = c.lower().replace("_", "").replace(" ", "")
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")


def parse_fets_partition_csv(csv_path: str | Path) -> Dict[str, str]:
    """Parse a FeTS partition CSV into a mapping of {subject_id: partition_id}."""
    df = pd.read_csv(csv_path)
    subj_col = find_column(df, ["subject_id", "subjectid", "subject", "id", "fets_id"])
    part_col = find_column(df, ["partition", "partition_id", "institution", "center", "client"])

    mapping = {}
    for _, row in df.iterrows():
        sid = str(row[subj_col]).strip()
        pid = str(row[part_col]).strip()
        mapping[sid] = pid
    return mapping


def make_partitions(
    metadata_csv: str | Path,
    source_data_dir: str | Path,
    output_root: str | Path,
    use_symlinks: bool = True,
) -> Dict[str, int]:
    """Organize FeTS source data into client directories:

        output_root/
          client_<id>/
            imagesTr/
            labelsTr/

    Uses symbolic links by default to avoid duplicating large MRI volumes.
    """
    mapping = parse_fets_partition_csv(metadata_csv)
    src_dir = Path(source_data_dir)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}

    for sid, pid in mapping.items():
        client_dir = out_dir / f"client_{pid}"
        img_dest = client_dir / "imagesTr"
        lbl_dest = client_dir / "labelsTr"
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        # Match subject files in source directory
        matched_files = list(src_dir.glob(f"**/{sid}*"))
        if not matched_files:
            continue

        for f in matched_files:
            target_folder = lbl_dest if ("seg" in f.name.lower() or "label" in f.name.lower()) else img_dest
            target_path = target_folder / f.name

            if not target_path.exists():
                try:
                    if use_symlinks:
                        os.symlink(f, target_path)
                    else:
                        shutil.copy2(f, target_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(f, target_path)

        counts[pid] = counts.get(pid, 0) + 1

    logger.info("Partitioned %d subjects across %d clients.", sum(counts.values()), len(counts))
    return counts
