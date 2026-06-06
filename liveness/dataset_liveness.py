"""dataset_liveness.py — LCC-FASD loader (độc lập, đuôi _liveness).

Tiền xử lý riêng cho liveness (KHÔNG dùng chung với deepfake): RGB, resize, ToTensor, normalize ImageNet.
Nhãn suy từ tên thư mục: real/live = 0, spoof/fake/attack = 1.
"""
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from model_liveness import IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LIVE_KEYS = ("real", "live", "genuine", "bonafide", "client")
SPOOF_KEYS = ("spoof", "fake", "attack", "print", "replay", "photo", "imposter")


def _infer_label_liveness(path):
    parts = path.replace("\\", "/").lower().split("/")
    for seg in reversed(parts[:-1]):
        if any(k in seg for k in SPOOF_KEYS):
            return 1
        if any(k in seg for k in LIVE_KEYS):
            return 0
    fname = parts[-1]
    if any(k in fname for k in SPOOF_KEYS):
        return 1
    if any(k in fname for k in LIVE_KEYS):
        return 0
    return None


def list_split_liveness(root):
    """[(image_path, label)] dưới `root`, nhãn suy từ thư mục."""
    items = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if p.lower().endswith(IMG_EXT):
            lab = _infer_label_liveness(p)
            if lab is not None:
                items.append((p, lab))
    return items


def _build_aug_liveness():
    import albumentations as A
    try:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.4),
        ])
    except Exception as e:
        print(f"[aug_liveness] fallback flip-only ({e})")
        return A.Compose([A.HorizontalFlip(p=0.5)])


class LCCFASDLiveness(Dataset):
    def __init__(self, items, resolution=224, augment=False):
        self.items = items
        self.res = resolution
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
        self.aug = _build_aug_liveness() if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        bgr = cv2.imread(path)
        if bgr is None:
            import numpy as np
            rgb = (torch.zeros(self.res, self.res, 3).numpy() + 127).astype("uint8")
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]
        rgb = cv2.resize(rgb, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        x = self.normalize(self.to_tensor(rgb))
        return x, label
