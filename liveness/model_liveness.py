"""model_liveness.py — B4 baseline cho LIVENESS / Face Anti-Spoofing.

SELF-CONTAINED: dùng torchvision EfficientNet-B4 (ImageNet pretrained), KHÔNG import bất kỳ file
detector deepfake nào (training/detectors/...) -> không xung đột với nhánh deepfake.

Nhãn quy ước: live/real = 0, spoof/attack = 1 (spoof = lớp dương).
"""
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


def build_b4_liveness(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 + đầu phân loại 2 lớp live/spoof. Trả về nn.Module: forward(x)->logits[B,2]."""
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features            # 1792
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ImageNet normalization (torchvision B4 kỳ vọng đúng bộ này)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
