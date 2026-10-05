"""Portable DP decoder components, with matching operations and state-dict keys.

Adapted from ZhengYinan-AIR/Diffusion-Planner at
a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df:
  diffusion_planner/model/module/dit.py
  diffusion_planner/model/diffusion_planner.py (decoder initialization)

Only the dense, zero-dropout subset of timm Mlp used by DP is needed here.
Keeping it in PyTorch lets cached-feature training run without importing the
planner, timm, torchvision or nuPlan. Optional parity tests execute the official
source with real timm and strictly load these components' state dictionaries.
"""

from torch import nn


class Mlp(nn.Module):
    """The dense timm Mlp path used by the official decoder (drop=0)."""

    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features)
        self.drop2 = nn.Dropout(0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Official DP order, including non-residual cross-attention and second MLP."""

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp1 = Mlp(dim, mlp_hidden_dim, act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, mlp_hidden_dim, act_layer=approx_gelu)

    def forward(self, x, cross_c, y, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y
        ).chunk(6, dim=1)
        z = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(z, z, z, key_padding_mask=attn_mask)[0]
        z = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(z)
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        return self.mlp2(self.norm4(x))


class FinalLayer(nn.Module):
    """Official adaptive normalization and two-layer output projection."""

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        return self.proj(modulate(self.norm_final(x), shift, scale))


def initialize_dit_weights(model):
    """DP decoder initialization, excluding its absent diffusion-time embedding."""

    def basic_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    model.apply(basic_init)
    for block in model.blocks:
        nn.init.zeros_(block.adaLN_modulation[-1].weight)
        nn.init.zeros_(block.adaLN_modulation[-1].bias)
    nn.init.zeros_(model.final_layer.adaLN_modulation[-1].weight)
    nn.init.zeros_(model.final_layer.adaLN_modulation[-1].bias)
    nn.init.zeros_(model.final_layer.proj[-1].weight)
    nn.init.zeros_(model.final_layer.proj[-1].bias)
