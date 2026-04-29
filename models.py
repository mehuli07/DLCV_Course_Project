import math
import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import VIT_CFG, DATASET_CFG

#  Vision Transformer

class PatchEmbed(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size:    int   = 32,
        patch_size:  int   = 4,
        in_chans:    int   = 3,
        num_classes: int   = 10,
        embed_dim:   int   = 192,
        depth:       int   = 9,
        num_heads:   int   = 3,
        mlp_ratio:   float = 4.0,
        drop_rate:   float = 0.1,
        attn_drop:   float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)    
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # parameter groups 

    def _is_norm(self, name: str) -> bool:
        return "norm" in name        

    def norm_params(self):
        for name, param in self.named_parameters():
            if self._is_norm(name):
                yield name, param

    def other_params(self):
        for name, param in self.named_parameters():
            if not self._is_norm(name):
                yield name, param

    # forward 

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


#  ResNet-50

class ResNet50(nn.Module):

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def _is_norm(self, name: str) -> bool:
        return "bn" in name or "downsample.1" in name

    def norm_params(self):
        for name, param in self.named_parameters():
            if self._is_norm(name):
                yield name, param

    def other_params(self):
        for name, param in self.named_parameters():
            if not self._is_norm(name):
                yield name, param

    def forward(self, x):
        return self.backbone(x)


#  Factory

def build_model(arch: str, dataset: str) -> nn.Module:
    num_classes = DATASET_CFG[dataset]["num_classes"]
    img_size = DATASET_CFG[dataset]["img_size"]

    if arch == "vit":
        return VisionTransformer(
            img_size = img_size,
            num_classes = num_classes,
            **VIT_CFG,
        )
    elif arch == "resnet50":
        return ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
