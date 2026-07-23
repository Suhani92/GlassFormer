"""GlassFormer: SegFormer-B2 backbone with cross-modal RadarAttention.

The RGB image is encoded by a pure SegFormer-B2 MiT encoder. The radar-derived
transparency prior is injected at the two deepest encoder stages (H/16 and H/32)
via a lightweight cross-modal attention module (RadarAttention), where the radar
embedding provides the query and the RGB features provide key/value. A learnable
gate ``gamma`` controls how strongly the attention output modulates the RGB
features, letting the network reject noisy radar priors.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


class RadarAttention(nn.Module):
    """Cross-modal attention that modulates RGB features with the radar prior.

    Args:
        feat_dim: Channel dimension of the backbone feature map.
        radar_dim: Hidden dimension of the 2-layer conv encoder for the radar mask.
    """

    def __init__(self, feat_dim, radar_dim=64):
        super().__init__()

        self.radar_encoder = nn.Sequential(
            nn.Conv2d(1, radar_dim, 3, padding=1),
            nn.BatchNorm2d(radar_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(radar_dim, radar_dim, 3, padding=1),
            nn.BatchNorm2d(radar_dim),
            nn.ReLU(inplace=True),
        )

        self.radar_proj = nn.Conv2d(radar_dim, feat_dim, 1)

        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)

        self.scale = feat_dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(0.1))  # learnable spatial gate (small init)

    def forward(self, feat, radar_mask):
        """
        Args:
            feat: RGB backbone features, shape (B, C, H, W).
            radar_mask: Radar transparency prior, shape (B, 1, H0, W0).

        Returns:
            Feature map of shape (B, C, H, W) with the radar prior fused in.
        """
        B, C, H, W = feat.shape

        radar_resized = F.interpolate(
            radar_mask, size=(H, W), mode="bilinear", align_corners=False
        )

        radar_feat = self.radar_encoder(radar_resized)
        radar_feat = self.radar_proj(radar_feat)

        feat_flat = feat.flatten(2).transpose(1, 2)        # (B, HW, C)
        radar_flat = radar_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        Q = self.query(radar_flat)
        K = self.key(feat_flat)
        V = self.value(feat_flat)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return feat + self.gamma * out


class GlassSegFormerRGBRadar(nn.Module):
    """SegFormer-B2 with RadarAttention at the two deepest stages.

    Args:
        pretrained: HuggingFace model id for the SegFormer backbone weights.
    """

    def __init__(self, pretrained="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()

        # Backbone stays pure-RGB; radar enters only through RadarAttention.
        self.encoder = SegformerModel.from_pretrained(pretrained)

        hidden_sizes = self.encoder.config.hidden_sizes

        # RadarAttention only on the two deepest stages (stages 3 and 4).
        # Stages 1 and 2 are left unmodified (attention cost > benefit there).
        self.radar_attn = nn.ModuleList([
            None,
            None,
            RadarAttention(hidden_sizes[2]),
            RadarAttention(hidden_sizes[3]),
        ])

        # Per-stage decoder projections to a common channel dim.
        self.mlps = nn.ModuleList([nn.Linear(h, 256) for h in hidden_sizes])

        self.fuse = nn.Conv2d(256 * 4, 256, 1)
        self.head = nn.Conv2d(256, 1, 1)

    def forward(self, rgb, radar_mask):
        outputs = self.encoder(
            pixel_values=rgb,
            output_hidden_states=True,
            return_dict=True,
        )

        feats = outputs.hidden_states
        processed_feats = []

        # Use the first (highest-resolution) stage as the decoder reference size.
        target_size = feats[0].shape[-2:]

        for i, f in enumerate(feats):
            if self.radar_attn[i] is not None:
                f = self.radar_attn[i](f, radar_mask)

            B, C, H, W = f.shape

            f = f.flatten(2).transpose(1, 2)
            f = self.mlps[i](f)
            f = f.transpose(1, 2).reshape(B, 256, H, W)

            f = F.interpolate(
                f, size=target_size, mode="bilinear", align_corners=False
            )
            processed_feats.append(f)

        x = torch.cat(processed_feats, dim=1)
        x = self.fuse(x)

        # Upsample back to the input resolution.
        x = F.interpolate(
            x, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )

        return self.head(x)


if __name__ == "__main__":
    model = GlassSegFormerRGBRadar()
    rgb = torch.randn(2, 3, 402, 402)
    radar = torch.randn(2, 1, 402, 402)
    out = model(rgb, radar)
    print("Output shape:", out.shape)
