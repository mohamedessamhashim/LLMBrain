"""UCSF-PDGM dataset loader for brain tumor segmentation.

Supports two data layouts:

1. **Raw** (directly from TCIA download, before preprocessing):
   ```
   data_dir/
   ├── UCSF-PDGM-0001/
   │   ├── *_t1.nii.gz
   │   ├── *_t1ce.nii.gz  (or *_t1gd.nii.gz)
   │   ├── *_t2.nii.gz
   │   ├── *_flair.nii.gz
   │   └── *_seg.nii.gz
   ├── UCSF-PDGM-0002/
   │   └── ...
   ```

2. **Preprocessed** (after running scripts/preprocess.py):
   ```
   data_dir/
   ├── images/
   │   ├── UCSF-PDGM-0001_0000.nii.gz  # T1
   │   ├── UCSF-PDGM-0001_0001.nii.gz  # T1ce
   │   ├── UCSF-PDGM-0001_0002.nii.gz  # T2
   │   └── UCSF-PDGM-0001_0003.nii.gz  # FLAIR
   └── labels/
       └── UCSF-PDGM-0001.nii.gz
   ```

The layout is auto-detected.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .transforms import get_train_transforms, get_val_transforms

DEFAULT_PROMPT = "brain MRI of a patient with diffuse glioma"

# Filename patterns used to discover modality files inside a raw subject dir.
# Order matters: earlier patterns are tried first; for T1 we explicitly
# exclude t1ce/t1gd so we don't accidentally match the contrast-enhanced scan.
_RAW_MODALITY_PATTERNS = {
    "t1": [
        "*_t1.nii.gz", "*_T1.nii.gz", "*_t1w.nii.gz", "*_T1W.nii.gz",
    ],
    "t1ce": [
        "*_t1ce.nii.gz", "*_T1CE.nii.gz",
        "*_t1gd.nii.gz", "*_T1GD.nii.gz",
        "*_t1Gd.nii.gz",
    ],
    "t2": [
        "*_t2.nii.gz", "*_T2.nii.gz", "*_t2w.nii.gz", "*_T2W.nii.gz",
    ],
    "flair": [
        "*_flair.nii.gz", "*_FLAIR.nii.gz", "*_Flair.nii.gz",
    ],
    "seg": [
        "*_seg.nii.gz", "*_SEG.nii.gz",
        "*_tumor_segmentation.nii.gz",
        "*_mask.nii.gz",
    ],
}


def _find_raw_modality(subject_dir: Path, modality: str) -> Optional[Path]:
    """Find a modality file inside a raw subject directory."""
    for pattern in _RAW_MODALITY_PATTERNS.get(modality, []):
        matches = list(subject_dir.glob(pattern))
        if modality == "t1":
            # Exclude t1ce / t1gd matches
            matches = [
                m for m in matches
                if "t1ce" not in m.name.lower()
                and "t1gd" not in m.name.lower()
                and "t1Gd" not in m.name
            ]
        if matches:
            return matches[0]
    return None


def _detect_layout(data_dir: Path) -> str:
    """Auto-detect whether *data_dir* uses the raw or preprocessed layout."""
    if (data_dir / "images").is_dir() and (data_dir / "labels").is_dir():
        return "preprocessed"
    # Check if there are sub-directories that look like subject folders
    for child in data_dir.iterdir():
        if child.is_dir() and child.name.startswith("UCSF-PDGM"):
            return "raw"
    # Fallback: if there are subject dirs without the prefix, still treat as raw
    for child in data_dir.iterdir():
        if child.is_dir() and any(child.glob("*.nii.gz")):
            return "raw"
    return "preprocessed"


class UCSFPDGMDataset:
    """UCSF-PDGM dataset wrapper.

    Auto-detects raw vs preprocessed layout.  When using the raw layout,
    MONAI transforms handle resampling to 1 mm isotropic and z-score
    normalisation at load time -- no separate preprocessing step needed.

    Args:
        data_dir: Root directory containing the data (raw *or* preprocessed).
        prompts_csv: Path to generated prompts CSV.
    """

    def __init__(
        self,
        data_dir: str | Path,
        prompts_csv: Optional[str | Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.layout = _detect_layout(self.data_dir)

        # Load prompts if provided
        self.prompts = None
        if prompts_csv is not None and Path(prompts_csv).exists():
            self.prompts = pd.read_csv(prompts_csv)

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def get_subject_ids(self) -> List[str]:
        if self.layout == "preprocessed":
            labels_dir = self.data_dir / "labels"
            return sorted(
                f.stem.replace(".nii", "") for f in labels_dir.glob("*.nii.gz")
            )
        else:
            # Raw: each sub-directory is a subject
            ids = []
            for child in sorted(self.data_dir.iterdir()):
                if child.is_dir() and any(child.glob("*.nii.gz")):
                    ids.append(child.name)
            return ids

    # ------------------------------------------------------------------
    # Build data list
    # ------------------------------------------------------------------

    def _entry_preprocessed(self, subject_id: str) -> Optional[Dict]:
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        image_paths = [
            str(images_dir / f"{subject_id}_000{i}.nii.gz") for i in range(4)
        ]
        if not all(Path(p).exists() for p in image_paths):
            return None

        label_path = labels_dir / f"{subject_id}.nii.gz"
        if not label_path.exists():
            return None

        return {"image": image_paths, "label": str(label_path)}

    def _entry_raw(self, subject_id: str) -> Optional[Dict]:
        subject_dir = self.data_dir / subject_id

        # Find each modality
        paths = {}
        for mod in ("t1", "t1ce", "t2", "flair"):
            p = _find_raw_modality(subject_dir, mod)
            if p is None:
                return None
            paths[mod] = p

        seg = _find_raw_modality(subject_dir, "seg")
        if seg is None:
            return None

        # Order: T1, T1ce, T2, FLAIR (channels 0-3)
        image_paths = [
            str(paths["t1"]),
            str(paths["t1ce"]),
            str(paths["t2"]),
            str(paths["flair"]),
        ]
        return {"image": image_paths, "label": str(seg)}

    def create_data_list(
        self, subject_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Build list of dicts for MONAI data loaders.

        Each dict: ``{image, label, subject_id, prompt}``.
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        builder = (
            self._entry_preprocessed
            if self.layout == "preprocessed"
            else self._entry_raw
        )

        data_list = []
        for sid in subject_ids:
            entry = builder(sid)
            if entry is None:
                continue

            # Attach prompt
            prompt = DEFAULT_PROMPT
            if self.prompts is not None:
                row = self.prompts[self.prompts["subject_id"] == sid]
                if not row.empty:
                    prompt = row.iloc[0]["prompt"]

            entry["subject_id"] = sid
            entry["prompt"] = prompt
            data_list.append(entry)

        return data_list


# ======================================================================
# Collate helper
# ======================================================================

def prompt_collate_fn(batch):
    """Collate that keeps string fields (prompt, subject_id) as lists.

    Also handles the case where each dataset item is a list of dicts
    (produced by RandCropByPosNegLabeld with num_samples > 1) by
    flattening the nested list before collation.
    """
    # Flatten list-of-lists: RandCropByPosNegLabeld(num_samples=N) returns
    # a list of N dicts per __getitem__ call; DataLoader batches those lists.
    flat: list = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    batch = flat

    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], (str, list)):
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


# ======================================================================
# Convenience loader factory
# ======================================================================

def get_train_val_loaders(
    data_dir: str | Path,
    prompts_csv: Optional[str | Path] = None,
    val_split: float = 0.2,
    batch_size: int = 2,
    num_workers: int = 4,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    cache_rate: float = 0.0,
    random_state: int = 42,
    # kept for backwards compat -- ignored
    metadata_csv: Optional[str | Path] = None,
    processed_dir: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        data_dir: Root data directory (raw or preprocessed).
        prompts_csv: Path to prompts CSV.
        val_split: Fraction for validation.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        img_size: Patch size for random crops during training.
        cache_rate: MONAI CacheDataset cache rate.
        random_state: Seed for train/val split.

    Returns:
        ``(train_loader, val_loader)``
    """
    # Backwards compat: if caller passes processed_dir, use that
    if data_dir is None and processed_dir is not None:
        data_dir = processed_dir
    if data_dir is None:
        raise ValueError("data_dir is required")

    dataset = UCSFPDGMDataset(data_dir, prompts_csv)
    data_list = dataset.create_data_list()

    if len(data_list) == 0:
        raise ValueError(
            f"No valid data found in {data_dir} (detected layout: {dataset.layout})"
        )

    print(f"Data layout: {dataset.layout}")
    print(f"Total subjects: {len(data_list)}")

    train_data, val_data = train_test_split(
        data_list, test_size=val_split, random_state=random_state
    )
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    train_transforms = get_train_transforms(img_size=img_size)
    val_transforms = get_val_transforms()

    if cache_rate > 0:
        train_ds = CacheDataset(
            data=train_data, transform=train_transforms,
            cache_rate=cache_rate, num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_data, transform=val_transforms,
            cache_rate=cache_rate, num_workers=num_workers,
        )
    else:
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=prompt_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=prompt_collate_fn,
    )

    return train_loader, val_loader
