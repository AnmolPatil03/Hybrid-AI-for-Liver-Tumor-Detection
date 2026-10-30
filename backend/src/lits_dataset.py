"""
LiTS (Liver Tumor Segmentation) PNG dataset: indexing, volume-level
splitting and 2.5D PyTorch datasets.

Expected layout (Kaggle: andrewmvd/lits-png), all files in one flat folder:

    <root>/
        volume-{vol}_{slice}.png                    CT slice
        segmentation-{vol}_livermask_{slice}.png    liver mask, binary {0,1}
        segmentation-{vol}_lesionmask_{slice}.png   tumor mask, binary {0,1}

The full dataset is used: 131 volumes, 58,638 slices. Only the lesion
(tumor) mask is used as the training target.

Splits are made at the volume level (70/15/15). Adjacent slices of a CT
volume are almost identical, so splitting by slice would leak test data
into training.
"""

import os
import re
import json
import glob
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import torchvision.transforms as T
import torchvision.transforms.functional as TF

SEG_IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_ROOT = os.path.join("data", "lits_png", "dataset_6", "dataset_6")
DEFAULT_SPLIT_PATH = os.path.join("outputs", "splits", "segmentation_split.json")

_VOLUME_PATTERN = re.compile(r"^volume-(\d+)_(\d+)\.png$", re.IGNORECASE)
_LESIONMASK_PATTERN = re.compile(r"^segmentation-(\d+)_lesionmask_(\d+)\.png$", re.IGNORECASE)


def extract_volume_and_slice(filename, pattern=None):
    """Return (volume_id, slice_idx) from a LiTS PNG filename."""
    base = os.path.basename(filename)
    for pat in (pattern,) if pattern else (_VOLUME_PATTERN, _LESIONMASK_PATTERN):
        m = pat.search(base)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise ValueError(
        f"Could not parse volume/slice ID from filename '{base}'. Expected "
        f"'volume-{{id}}_{{slice}}.png' or 'segmentation-{{id}}_lesionmask_{{slice}}.png'."
    )


def summarize_mask_values(root=DEFAULT_ROOT, n=5):
    """Print unique pixel values of a few lesion masks (quick format check)."""
    paths = sorted(glob.glob(os.path.join(root, "segmentation-*_lesionmask_*.png")))[:n]
    if not paths:
        print(f"[WARN] No lesion mask PNGs found under '{root}'.")
        return
    for p in paths:
        arr = np.array(PILImage.open(p))
        print(f"{os.path.basename(p)}: unique values = {sorted(np.unique(arr).tolist())}")


def build_sample_index(root=DEFAULT_ROOT):
    """
    Pair every CT slice with its lesion mask.

    Returns a list of dicts: {volume_id, slice_idx, image_path, mask_path}.
    """
    vol_paths = sorted(glob.glob(os.path.join(root, "volume-*.png")))
    mask_paths = sorted(glob.glob(os.path.join(root, "segmentation-*_lesionmask_*.png")))

    mask_lookup = {extract_volume_and_slice(mp, _LESIONMASK_PATTERN): mp for mp in mask_paths}

    samples = []
    skipped = 0
    for vp in vol_paths:
        key = extract_volume_and_slice(vp, _VOLUME_PATTERN)
        mp = mask_lookup.get(key)
        if mp is None:
            skipped += 1
            continue
        vol_id, slice_idx = key
        samples.append({
            "volume_id": vol_id, "slice_idx": slice_idx,
            "image_path": vp, "mask_path": mp,
        })

    if not samples:
        raise FileNotFoundError(
            f"No paired volume/lesion-mask PNG samples found under '{root}'. "
            f"Expected files like 'volume-0_0.png' and "
            f"'segmentation-0_lesionmask_0.png' directly inside that folder."
        )
    if skipped:
        print(f"[WARN] {skipped} slice(s) had no matching lesion mask and were skipped.")

    print(f"[INFO] Indexed {len(samples)} paired slices across "
          f"{len(set(s['volume_id'] for s in samples))} volumes.")
    return samples


def _volume_has_tumor(slices):
    for s in slices:
        if np.any(np.array(PILImage.open(s["mask_path"])) > 0):
            return True
    return False


def split_by_volume(samples, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                    test_ratio=TEST_RATIO, stratify_by_tumor=True, split_path=DEFAULT_SPLIT_PATH):
    """
    Split whole volumes into train/val/test, optionally stratified by whether
    a volume contains any tumor. Saves the mapping to `split_path`.

    Returns (train_samples, val_samples, test_samples, mapping).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Split ratios must sum to 1.0"

    volume_slices = defaultdict(list)
    for s in samples:
        volume_slices[s["volume_id"]].append(s)
    volume_ids = sorted(volume_slices)

    if stratify_by_tumor:
        tumor_positive = {vid: _volume_has_tumor(volume_slices[vid]) for vid in volume_ids}
        pos_ids = [v for v in volume_ids if tumor_positive[v]]
        neg_ids = [v for v in volume_ids if not tumor_positive[v]]
    else:
        pos_ids, neg_ids = volume_ids, []

    rng = random.Random(seed)

    def split_group(ids):
        ids = ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    train_pos, val_pos, test_pos = split_group(pos_ids)
    train_neg, val_neg, test_neg = split_group(neg_ids)

    mapping = {
        "train": sorted(train_pos + train_neg),
        "val": sorted(val_pos + val_neg),
        "test": sorted(test_pos + test_neg),
        "seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio,
        "test_ratio": test_ratio, "stratify_by_tumor": stratify_by_tumor,
    }
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Volume split saved to: {split_path}")

    return (*_apply_split(samples, mapping), mapping)


def _apply_split(samples, mapping):
    ids = {k: set(mapping[k]) for k in ("train", "val", "test")}
    train = [s for s in samples if s["volume_id"] in ids["train"]]
    val = [s for s in samples if s["volume_id"] in ids["val"]]
    test = [s for s in samples if s["volume_id"] in ids["test"]]
    print(f"[INFO] Volumes - train:{len(ids['train'])} val:{len(ids['val'])} test:{len(ids['test'])}")
    print(f"[INFO] Slices  - train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test


def load_or_create_split(samples, split_path=DEFAULT_SPLIT_PATH, seed=SEED, stratify_by_tumor=True):
    """
    Reuse the saved volume split if it exists so every run (and the reported
    test metrics) uses the same held-out volumes; otherwise create it.
    """
    if os.path.exists(split_path):
        with open(split_path) as f:
            mapping = json.load(f)
        print(f"[INFO] Using existing volume split: {split_path}")
        return (*_apply_split(samples, mapping), mapping)
    return split_by_volume(samples, seed=seed, stratify_by_tumor=stratify_by_tumor, split_path=split_path)


def verify_split_integrity(split_path=DEFAULT_SPLIT_PATH, root=DEFAULT_ROOT):
    """
    Assert that no volume appears in more than one split and print counts.

        python -c "from src.lits_dataset import verify_split_integrity; verify_split_integrity()"
    """
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"No split mapping at '{split_path}'. Run split_by_volume() first.")
    with open(split_path) as f:
        mapping = json.load(f)

    train_ids, val_ids, test_ids = set(mapping["train"]), set(mapping["val"]), set(mapping["test"])
    assert not train_ids & val_ids, f"Volumes leaked between train/val: {train_ids & val_ids}"
    assert not train_ids & test_ids, f"Volumes leaked between train/test: {train_ids & test_ids}"
    assert not val_ids & test_ids, f"Volumes leaked between val/test: {val_ids & test_ids}"
    print("[OK] No volume appears in more than one split.")

    samples = build_sample_index(root)
    train, val, test = _apply_split(samples, mapping)
    return {
        "volumes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "slices": {"train": len(train), "val": len(val), "test": len(test)},
    }


class LiTSSeg25DDataset(Dataset):
    """
    2.5D LiTS dataset: the slices at [-1, 0, +1] around the target slice are
    stacked as the 3 input channels. This keeps the (3, H, W) input that the
    ImageNet-pretrained ConvNeXt stem expects while giving the model some
    inter-slice context. At the first/last slice of a volume the centre
    slice is repeated.

    With `augment=True`, the same random flip / rotation / brightness /
    contrast / elastic deformation is applied to all three channels and the
    mask (nearest-neighbour for the mask so it stays binary).
    """

    NEIGHBOR_OFFSETS = (-1, 0, 1)

    def __init__(self, samples, root, augment=False,
                 elastic_alpha=20.0, elastic_sigma=4.0, elastic_prob=0.5):
        self.samples = samples
        self.root = root
        self.augment = augment
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

        self._volume_max_index = defaultdict(int)
        for s in samples:
            vid = s["volume_id"]
            self._volume_max_index[vid] = max(self._volume_max_index[vid], s["slice_idx"])

    def __len__(self):
        return len(self.samples)

    def _neighbor_path(self, sample, slice_idx):
        if slice_idx == sample["slice_idx"]:
            return sample["image_path"]
        return os.path.join(self.root, f"volume-{sample['volume_id']}_{slice_idx}.png")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        max_idx = self._volume_max_index[sample["volume_id"]]

        slice_imgs = []
        for off in self.NEIGHBOR_OFFSETS:
            n_idx = min(max(sample["slice_idx"] + off, 0), max_idx)
            img = PILImage.open(self._neighbor_path(sample, n_idx)).convert("L")
            slice_imgs.append(img.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE), PILImage.BILINEAR))

        mask = PILImage.open(sample["mask_path"])
        mask = mask.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE), PILImage.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                slice_imgs = [TF.hflip(im) for im in slice_imgs]
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                slice_imgs = [TF.vflip(im) for im in slice_imgs]
                mask = TF.vflip(mask)
            angle = random.uniform(-15, 15)
            slice_imgs = [TF.rotate(im, angle, interpolation=TF.InterpolationMode.BILINEAR) for im in slice_imgs]
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            brightness = 1.0 + random.uniform(-0.1, 0.1)
            contrast = 1.0 + random.uniform(-0.1, 0.1)
            slice_imgs = [TF.adjust_contrast(TF.adjust_brightness(im, brightness), contrast) for im in slice_imgs]

        image = torch.cat([TF.to_tensor(im) for im in slice_imgs], dim=0)  # (3, H, W)

        mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
        tumor_mask = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)  # (1, H, W)

        if self.augment and random.random() < self.elastic_prob:
            displacement = T.ElasticTransform.get_params(
                alpha=[self.elastic_alpha, self.elastic_alpha],
                sigma=[self.elastic_sigma, self.elastic_sigma],
                size=[SEG_IMAGE_SIZE, SEG_IMAGE_SIZE],
            )
            image = TF.elastic_transform(image, displacement, interpolation=TF.InterpolationMode.BILINEAR)
            tumor_mask = TF.elastic_transform(tumor_mask, displacement, interpolation=TF.InterpolationMode.NEAREST)
            tumor_mask = (tumor_mask > 0.5).float()

        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, tumor_mask


def _seed_worker(worker_id):
    """Give each DataLoader worker its own deterministic RNG seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lits_dataloaders_25d(root=DEFAULT_ROOT, split_path=DEFAULT_SPLIT_PATH,
                             batch_size=16, num_workers=0, seed=None,
                             elastic_alpha=20.0, elastic_sigma=4.0, elastic_prob=0.5):
    """
    Build 2.5D train/val/test DataLoaders over the full LiTS dataset using
    the volume-level split.

    Returns (train_loader, val_loader, test_loader, dataset_info).
    """
    samples = build_sample_index(root)
    train_samples, val_samples, test_samples, mapping = load_or_create_split(
        samples, split_path=split_path, seed=seed if seed is not None else SEED,
    )

    # Neighbour lookup needs the last slice index of every volume, so each
    # dataset gets the samples of its own volumes only (no cross-split reads).
    train_dataset = LiTSSeg25DDataset(
        train_samples, root=root, augment=True,
        elastic_alpha=elastic_alpha, elastic_sigma=elastic_sigma, elastic_prob=elastic_prob,
    )
    val_dataset = LiTSSeg25DDataset(val_samples, root=root, augment=False)
    test_dataset = LiTSSeg25DDataset(test_samples, root=root, augment=False)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    worker_init = _seed_worker if seed is not None else None
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              worker_init_fn=worker_init, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, worker_init_fn=worker_init)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin, worker_init_fn=worker_init)

    dataset_info = {
        "total_slices": len(samples),
        "train_slices": len(train_samples), "val_slices": len(val_samples), "test_slices": len(test_samples),
        "train_volumes": len(mapping["train"]), "val_volumes": len(mapping["val"]),
        "test_volumes": len(mapping["test"]),
        "image_size": SEG_IMAGE_SIZE,
    }
    return train_loader, val_loader, test_loader, dataset_info
