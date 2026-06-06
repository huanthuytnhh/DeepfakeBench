"""LCC-FASD dataset for FAS fine-tuning.

Reuses the EXACT DeepfakeBench preprocessing so the existing B4 / SFDCT detectors accept it
unchanged: cv2 BGR->RGB, resize to `resolution` with INTER_CUBIC, ToTensor [0,1],
Normalize(mean,std) (0.5/0.5 -> [-1,1]). Label: live/real = 0, spoof/attack = 1.

Label is inferred from the folder name (LCC-FASD stores real/ and spoof/ subdirs inside each of
LCC_FASD_training / _development / _evaluation). list_split() prints nothing; the caller logs counts.
"""
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LIVE_KEYS = ("real", "live", "genuine", "bonafide", "bona_fide", "client")
SPOOF_KEYS = ("spoof", "fake", "attack", "print", "replay", "photo", "imposter")


def _infer_label(path):
    """0=live, 1=spoof, None=unknown. Checks folder segments (deepest first), then filename."""
    parts = path.replace("\\", "/").lower().split("/")
    for seg in reversed(parts[:-1]):            # directory names first (the class folder)
        if any(k in seg for k in SPOOF_KEYS):
            return 1
        if any(k in seg for k in LIVE_KEYS):
            return 0
    fname = parts[-1]                            # fall back to filename
    if any(k in fname for k in SPOOF_KEYS):
        return 1
    if any(k in fname for k in LIVE_KEYS):
        return 0
    return None


def list_split(root):
    """Return a sorted list of (image_path, label) under `root` (label inferred from path)."""
    items = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not p.lower().endswith(IMG_EXT):
            continue
        lab = _infer_label(p)
        if lab is not None:
            items.append((p, lab))
    return items


def _build_aug(aug_cfg, resolution):
    """Mirror the repo's train augmentation (albumentations). Falls back to flip+rotate if the
    installed albumentations version rejects a kwarg."""
    import albumentations as A
    c = aug_cfg or {}
    try:
        return A.Compose([
            A.HorizontalFlip(p=c.get("flip_prob", 0.5)),
            A.Rotate(limit=c.get("rotate_limit", [-10, 10]), p=c.get("rotate_prob", 0.5)),
            A.GaussianBlur(blur_limit=c.get("blur_limit", [3, 7]), p=c.get("blur_prob", 0.5)),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=c.get("brightness_limit", [-0.1, 0.1]),
                                           contrast_limit=c.get("contrast_limit", [-0.1, 0.1])),
                A.FancyPCA(),
                A.HueSaturationValue(),
            ], p=0.5),
            A.ImageCompression(quality_lower=c.get("quality_lower", 40),
                               quality_upper=c.get("quality_upper", 100), p=0.5),
        ])
    except Exception as e:                       # albumentations API drift -> minimal safe aug
        print(f"[aug] full pipeline failed ({e}); using flip+rotate only")
        return A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5)])


class LCCFASD(Dataset):
    def __init__(self, items, resolution=256, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 augment=False, aug_cfg=None):
        self.items = items
        self.res = resolution
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=list(mean), std=list(std))
        self.aug = _build_aug(aug_cfg, resolution) if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        bgr = cv2.imread(path)
        if bgr is None:                          # unreadable -> neutral gray (rare)
            import numpy as np
            rgb = (torch.zeros(self.res, self.res, 3).numpy() + 127).astype("uint8")
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]
        rgb = cv2.resize(rgb, (self.res, self.res), interpolation=cv2.INTER_CUBIC)
        x = self.normalize(self.to_tensor(rgb))  # [3,res,res], float, [-1,1]
        return x, label
