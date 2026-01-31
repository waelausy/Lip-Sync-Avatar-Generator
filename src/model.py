"""
Modèle Audio-conditioned U-Net Lite pour HygieSync
Conçu pour inférence rapide sur GPU 4GB / CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMB_DIM


def conv_block(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
    """Bloc convolutionnel avec BatchNorm et ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class AudioEncoder(nn.Module):
    """Encodeur audio: Mel spectrogram -> embedding"""
    
    def __init__(self, emb: int = EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, emb)

    def forward(self, a):
        x = self.net(a).flatten(1)
        return self.fc(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation pour conditionnement audio"""
    
    def __init__(self, emb: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb, channels)
        self.to_beta = nn.Linear(emb, channels)

    def forward(self, feat, aemb):
        g = self.to_gamma(aemb)[:, :, None, None]
        b = self.to_beta(aemb)[:, :, None, None]
        return feat * (1.0 + g) + b


class HygieUNetLite(nn.Module):
    """U-Net léger avec conditionnement audio via FiLM"""
    
    def __init__(self, emb: int = EMB_DIM):
        super().__init__()
        self.aenc = AudioEncoder(emb=emb)

        self.e1 = conv_block(3, 32)
        self.e2 = conv_block(32, 64, s=2)
        self.e3 = conv_block(64, 128, s=2)
        self.e4 = conv_block(128, 256, s=2)

        self.b = conv_block(256, 256)

        self.f4 = FiLM(emb, 256)
        self.d3 = conv_block(256 + 128, 128)
        self.f3 = FiLM(emb, 128)
        self.d2 = conv_block(128 + 64, 64)
        self.f2 = FiLM(emb, 64)
        self.d1 = conv_block(64 + 32, 32)
        self.f1 = FiLM(emb, 32)

        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x_masked, mel_chunk):
        aemb = self.aenc(mel_chunk)

        e1 = self.e1(x_masked)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.b(e4)

        u3 = F.interpolate(self.f4(b, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(self.f3(u3, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(self.f2(u2, aemb), scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, e1], dim=1))

        u1 = self.f1(u1, aemb)

        delta = torch.tanh(self.out(u1))
        yhat = torch.clamp(x_masked + delta, 0.0, 1.0)
        return yhat
