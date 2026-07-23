# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SegformerModel
# from torchview import draw_graph

# # -------------------------
# # Radar Cross Attention
# # -------------------------
# class RadarAttention(nn.Module):
#     def __init__(self, feat_dim, radar_dim=64):
#         super().__init__()

#         self.radar_encoder = nn.Sequential(
#             nn.Conv2d(1, radar_dim, 3, padding=1),
#             nn.BatchNorm2d(radar_dim),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(radar_dim, radar_dim, 3, padding=1),
#             nn.BatchNorm2d(radar_dim),
#             nn.ReLU(inplace=True),
#         )

#         self.radar_proj = nn.Conv2d(radar_dim, feat_dim, 1)

#         self.query = nn.Linear(feat_dim, feat_dim)
#         self.key   = nn.Linear(feat_dim, feat_dim)
#         self.value = nn.Linear(feat_dim, feat_dim)

#         self.scale = feat_dim ** -0.5
#         self.gamma = nn.Parameter(torch.tensor(0.1))  # small influence

#     def forward(self, feat, radar_mask):
#         """
#         feat: (B, C, H, W)
#         radar_mask: (B, 1, H0, W0)
#         """

#         B, C, H, W = feat.shape

#         radar_resized = F.interpolate(
#             radar_mask,
#             size=(H, W),
#             mode="bilinear",
#             align_corners=False
#         )

#         radar_feat = self.radar_encoder(radar_resized)
#         radar_feat = self.radar_proj(radar_feat)

#         feat_flat  = feat.flatten(2).transpose(1, 2)     # (B, HW, C)
#         radar_flat = radar_feat.flatten(2).transpose(1, 2)

#         Q = self.query(radar_flat)
#         K = self.key(feat_flat)
#         V = self.value(feat_flat)

#         attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#         attn = torch.softmax(attn, dim=-1)

#         out = torch.matmul(attn, V)
#         out = out.transpose(1, 2).reshape(B, C, H, W)

#         return feat + self.gamma * out


# # -------------------------
# # Main Model
# # -------------------------
# class GlassSegFormerRGBRadar(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # ✅ KEEP BACKBONE PURE RGB
#         self.encoder = SegformerModel.from_pretrained(
#             "nvidia/segformer-b2-finetuned-ade-512-512"
#         )

#         hidden_sizes = self.encoder.config.hidden_sizes

#         # Radar attention only on deeper stages
#         self.radar_attn = nn.ModuleList([
#             None,
#             None,
#             RadarAttention(hidden_sizes[2]),
#             RadarAttention(hidden_sizes[3])
#         ])

#         # Decoder projections
#         self.mlps = nn.ModuleList([
#             nn.Linear(h, 256) for h in hidden_sizes
#         ])

#         self.fuse = nn.Conv2d(256 * 4, 256, 1)
#         self.head = nn.Conv2d(256, 1, 1)

#     def forward(self, rgb, radar_mask):

#         #print("\n===== FORWARD PASS =====")
#         #print(f"Input RGB:     {list(rgb.shape)}")
#         #print(f"Input Radar:   {list(radar_mask.shape)}")

#         outputs = self.encoder(
#             pixel_values=rgb,
#             output_hidden_states=True,
#             return_dict=True
#         )

#         feats = outputs.hidden_states

#         #print("\n----- Encoder Features -----")
#         # for i, f in enumerate(feats):
#             #print(f"Encoder Feature {i}: {list(f.shape)}")

#         processed_feats = []
#         target_size = feats[0].shape[-2:]

#         for i, f in enumerate(feats):

#             if self.radar_attn[i] is not None:
#                 f = self.radar_attn[i](f, radar_mask)
#                 #print(f"Radar Attention Stage {i}: {list(f.shape)}")

#             B, C, H, W = f.shape

#             f = f.flatten(2).transpose(1, 2)
#             f = self.mlps[i](f)
#             f = f.transpose(1, 2).reshape(B, 256, H, W)

#             #print(f"After MLP {i}: {list(f.shape)}")

#             f = F.interpolate(
#                 f,
#                 size=target_size,
#                 mode="bilinear",
#                 align_corners=False
#             )

#             #print(f"Upsampled Feature {i}: {list(f.shape)}")

#             processed_feats.append(f)

#         #print("\n----- Decoder -----")

#         x = torch.cat(processed_feats, dim=1)
#         #print(f"Concat Features: {list(x.shape)}")

#         x = self.fuse(x)
#         #print(f"After Fuse:      {list(x.shape)}")

#         x = F.interpolate(
#             x,
#             size=rgb.shape[-2:],
#             mode="bilinear",
#             align_corners=False
#         )

#         #print(f"Upsampled Decoder: {list(x.shape)}")

#         mask = self.head(x)
#         #print(f"Final Mask:        {list(mask.shape)}")

#         #print("===== END FORWARD =====\n")

#         return mask
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


# -------------------------
# Radar Cross Attention
# -------------------------
class RadarAttention(nn.Module):
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
        self.key   = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)

        self.scale = feat_dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(0.1))  # small influence

    def forward(self, feat, radar_mask):
        """
        feat: (B, C, H, W)
        radar_mask: (B, 1, H0, W0)
        """

        B, C, H, W = feat.shape

        radar_resized = F.interpolate(
            radar_mask,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        radar_feat = self.radar_encoder(radar_resized)
        radar_feat = self.radar_proj(radar_feat)

        feat_flat  = feat.flatten(2).transpose(1, 2)     # (B, HW, C)
        radar_flat = radar_feat.flatten(2).transpose(1, 2)

        Q = self.query(radar_flat)
        K = self.key(feat_flat)
        V = self.value(feat_flat)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return feat + self.gamma * out


# -------------------------
# Main Model
# -------------------------
class GlassSegFormerRGBRadar(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ KEEP BACKBONE PURE RGB
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )

        hidden_sizes = self.encoder.config.hidden_sizes

        # Radar attention only on deeper stages
        self.radar_attn = nn.ModuleList([
            None,
            None,
            RadarAttention(hidden_sizes[2]),
            RadarAttention(hidden_sizes[3])
        ])

        # Decoder projections
        self.mlps = nn.ModuleList([
            nn.Linear(h, 256) for h in hidden_sizes
        ])

        self.fuse = nn.Conv2d(256 * 4, 256, 1)
        self.head = nn.Conv2d(256, 1, 1)

    def forward(self, rgb, radar_mask):

        # ✅ RGB only into backbone
        outputs = self.encoder(
            pixel_values=rgb,
            output_hidden_states=True,
            return_dict=True
        )

        feats = outputs.hidden_states
        processed_feats = []

        # Use first stage resolution as decoder reference
        target_size = feats[0].shape[-2:]

        for i, f in enumerate(feats):

            if self.radar_attn[i] is not None:
                f = self.radar_attn[i](f, radar_mask)

            B, C, H, W = f.shape

            f = f.flatten(2).transpose(1, 2)
            f = self.mlps[i](f)
            f = f.transpose(1, 2).reshape(B, 256, H, W)

            # ✅ No hardcoded sizes
            f = F.interpolate(
                f,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            processed_feats.append(f)

        x = torch.cat(processed_feats, dim=1)
        x = self.fuse(x)

        # Upsample back to input resolution
        x = F.interpolate(
            x,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        mask = self.head(x)

        return mask
    
if __name__ == "__main__":
    model = GlassSegFormerRGBRadar()

    rgb = torch.randn(2,3,402,402)
    radar = torch.randn(2,1,402,402)

    out = model(rgb, radar)
    # graph = draw_graph(
    #     model,
    #     input_data=(rgb, radar),
    #     expand_nested=True,
    #     depth=3
    # )

    # graph.visual_graph.render("model_architecture", format="png")
    print(out.shape)