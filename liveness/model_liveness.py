"""model_liveness.py — B4 baseline cho LIVENESS, làm TƯƠNG ĐƯƠNG NHẤT với detector B4 của DeepfakeBench.

Nhân bản đúng training/networks/efficientnetb4.py (mode=Original, dropout=False):
  - cùng backbone `efficientnet_pytorch` EfficientNet-B4 + CÙNG file pretrained (efficientnet-b4-6ed6700e.pth)
  - cùng thay `_conv_stem = Conv2d(inc,48,3,stride=2,bias=False)`, `_fc = Identity`
  - features = extract_features -> [B,1792,8,8]; classifier = GAP -> (dropout) -> Linear(1792,2)
  - CÙNG chuẩn hoá 0.5/0.5 và độ phân giải 256 (xem NORM_*/RESOLUTION).
SELF-CONTAINED: chỉ phụ thuộc pip `efficientnet_pytorch`, KHÔNG import training/detectors -> không xung đột.
Nhãn: live/real = 0, spoof/attack = 1.
"""
import os
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# KHỚP input contract của DeepfakeBench (KHÔNG phải ImageNet): normalize -> [-1,1], 256x256
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
RESOLUTION = 256

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEEPFAKE_PRETRAINED = os.path.join(_REPO, "training", "pretrained", "efficientnet-b4-6ed6700e.pth")


class B4Liveness(nn.Module):
    """Mirror byte-for-byte của EfficientNetB4 (DeepfakeBench) cho bài toán live/spoof."""

    def __init__(self, num_classes=2, inc=3, dropout=False, use_pretrained=True, pretrained_path=None):
        super().__init__()
        path = pretrained_path or (_DEEPFAKE_PRETRAINED if os.path.isfile(_DEEPFAKE_PRETRAINED) else None)
        if use_pretrained and path:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', weights_path=path)
        elif use_pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')   # tải weights lukemelas y hệt
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)  # GIỐNG detector
        self.efficientnet._fc = nn.Identity()
        self.dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.last_layer = nn.Linear(1792, num_classes)

    def features(self, x):
        return self.efficientnet.extract_features(x)              # [B,1792,8,8] @256

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        return self.last_layer(x)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_b4_liveness(num_classes=2, pretrained=True, pretrained_path=None):
    """API công khai: trả nn.Module forward(x)->logits[B,2]. Tương đương B4 detector deepfake."""
    return B4Liveness(num_classes=num_classes, inc=3, dropout=False,
                      use_pretrained=pretrained, pretrained_path=pretrained_path)
