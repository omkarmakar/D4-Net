
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Manual Toggle for Ablation
use_xception = True      # Toggle Xception branch
use_meso = True           # Toggle Meso branch
use_cross_att = True      # Only valid if both branches used
use_fusion = False        # Only valid if both branches used
dct_config = "none"       # Options: "none", "meso", "xception", "both"


def dct_transform(image_tensor):
    return torch.fft.fft2(image_tensor, dim=(-2, -1)).abs()


class MesoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2, padding=1),
            nn.Conv2d(8, 8, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2, padding=1),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2, padding=1), nn.Dropout(0.5),
            nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(4, padding=1), nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.conv(x)

class CrossAttention(nn.Module):
    def __init__(self, xcep_dim, meso_dim):
        super().__init__()
        self.query = nn.Linear(xcep_dim, xcep_dim)
        self.key = nn.Linear(meso_dim, xcep_dim)
        self.value = nn.Linear(meso_dim, xcep_dim)
        self.scale = xcep_dim ** 0.5

    def forward(self, xcep_feat, meso_feat):
        B, C1, H, W = xcep_feat.shape
        B, C2, _, _ = meso_feat.shape
        N = H * W

        q = xcep_feat.view(B, C1, N).permute(0, 2, 1)
        k = meso_feat.view(B, C2, N).permute(0, 2, 1)
        v = meso_feat.view(B, C2, N).permute(0, 2, 1)

        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        out = torch.bmm(attn, V)
        return out.permute(0, 2, 1).view(B, C1, H, W)

class AdaptiveFusionModule(nn.Module):
    def __init__(self, meso_in_channels=2048, xcep_in_channels=2048, out_channels=256):
        super().__init__()
        self.proj1 = nn.Conv2d(meso_in_channels, out_channels, 1)
        self.proj2 = nn.Conv2d(xcep_in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, meso_feat, xcep_feat):
        f1 = self.proj1(meso_feat)
        f2 = self.proj2(xcep_feat)
        return self.relu(self.bn(f1 + f2))
class DeepfakeHybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_xcep = use_xception
        self.use_meso = use_meso
        self.use_ca = use_cross_att and self.use_xcep and self.use_meso
        self.use_fusion = use_fusion and self.use_xcep and self.use_meso

        if self.use_xcep:
            self.xception = timm.create_model('xception', pretrained=True, features_only=True)
        if self.use_meso:
            self.meso_branch = MesoBlock()
        if self.use_ca:
            self.cross_att = CrossAttention(2048, 16)

        if self.use_fusion:
            meso_channels_into_fusion = 2048 if self.use_ca else 16
            self.fusion = AdaptiveFusionModule(
                meso_in_channels=meso_channels_into_fusion,
                xcep_in_channels=2048,
                out_channels=256
            )


        # Determine classifier input size
        if self.use_fusion:
            classifier_in = 256
        elif self.use_xcep and self.use_meso:
            classifier_in = 2048 + (2048 if self.use_ca else 16)  # concat fallback
        else:
            classifier_in = 2048 if self.use_xcep else 16

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        meso_feat, xcep_feat = None, None

        if self.use_xcep:
            xcep_in = dct_transform(x) if dct_config in ["xception", "both"] else x
            xcep_feat = self.xception(xcep_in)[-1]

        if self.use_meso:
            meso_in = dct_transform(x) if dct_config in ["meso", "both"] else x
            meso_feat = self.meso_branch(meso_in)

        if self.use_xcep and self.use_meso:
            if meso_feat.shape[-2:] != xcep_feat.shape[-2:]:
                meso_feat = F.interpolate(meso_feat, size=xcep_feat.shape[-2:], mode='bilinear', align_corners=False)

            if self.use_ca:
                meso_feat = self.cross_att(xcep_feat, meso_feat)

            if self.use_fusion:
                fused = self.fusion(meso_feat, xcep_feat)
            else:
                fused = torch.cat([xcep_feat, meso_feat], dim=1)  # Fallback merge
        else:
            fused = xcep_feat if self.use_xcep else meso_feat

        pooled = self.pool(fused)
        return self.classifier(pooled).squeeze(1)
