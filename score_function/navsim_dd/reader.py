"""Frozen DiffusionDrive scene processing and trajectory-dependent reading.

This is the prefix of the pinned DD first diffusion decoder layer, ending
before timestep modulation.  It has no sampler, anchors, regression head or
mode selection.  Plain Torch modules keep exported score checkpoints usable
without constructing the perception backbone or consulting the Hub.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from score_function.model.score_branch import ScoreFunctionBranch

DD_COMMIT = "9b52ed0ec06b073d82d6f392ab084c7b301c8681"
ASSET_FORMAT = "navsim-dd-reader-assets-v1"
ARCHITECTURE = "dd_scene_reader_score"


def _validate_spec(spec):
    required = {
        "latent_channels", "latent_shape", "d_model", "bev_channels", "bev_shape",
        "bev_upsample_factor", "num_agents", "scene_num_layers", "num_heads",
        "dim_feedforward", "scene_dropout", "lidar_max_x", "lidar_max_y", "num_poses",
    }
    if set(spec) != required:
        raise ValueError("Incomplete or unknown DD reader architecture fields")
    for key in ("latent_channels", "d_model", "bev_channels", "num_agents",
                "scene_num_layers", "num_heads", "dim_feedforward", "num_poses"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"Invalid reader {key}")
    for key in ("latent_shape", "bev_shape"):
        if len(spec[key]) != 2 or any(type(v) is not int or v <= 0 for v in spec[key]):
            raise ValueError(f"Invalid reader {key}")
    if spec["d_model"] != 256 or spec["num_poses"] != 8 or spec["d_model"] % spec["num_heads"]:
        raise ValueError("The frozen DD reader requires width 256 and eight poses")
    for key in ("bev_upsample_factor", "lidar_max_x", "lidar_max_y"):
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"Invalid reader {key}")
    if not 0 <= spec["scene_dropout"] < 1:
        raise ValueError("Invalid reader dropout")


def _trajectory_position_embedding(xy):
    """Exact gen_sineembed_for_position(xy, hidden_dim=64) from pinned DD."""
    dimensions = torch.arange(32, dtype=torch.float32, device=xy.device)
    dimensions = 10000 ** (2 * (dimensions // 2) / 32)
    x = xy[..., 0, None] * (2 * math.pi) / dimensions
    y = xy[..., 1, None] * (2 * math.pi) / dimensions
    x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
    y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((y, x), dim=-1)


class _GridSampleCrossBEVAttention(nn.Module):
    """The original learned eight-point aggregation, including its XY mapping."""

    def __init__(self, spec):
        super().__init__()
        width = spec["d_model"]
        self.attention_weights = nn.Linear(width, spec["num_poses"])
        self.output_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.1)
        self.value_proj = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.lidar_max_x = spec["lidar_max_x"]
        self.lidar_max_y = spec["lidar_max_y"]

    def forward(self, queries, physical_xy, value):
        # value is precomputed once per scene; the grid and aggregation change
        # with EVERY noisy training trajectory / current refinement iterate.
        grid = physical_xy.clone()
        grid[..., 0] = grid[..., 0] / self.lidar_max_y
        grid[..., 1] = grid[..., 1] / self.lidar_max_x
        grid = grid[..., [1, 0]]
        weights = self.attention_weights(queries).softmax(-1).unsqueeze(1)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)
        aggregated = (weights * sampled).sum(dim=-1).permute(0, 2, 1).contiguous()
        return self.dropout(self.output_proj(aggregated)) + queries


class FrozenDDReader(nn.Module):
    """All scene-only DD layers plus the first trajectory-conditioned prefix."""

    def __init__(self, spec):
        super().__init__()
        _validate_spec(spec)
        self.spec = copy.deepcopy(spec)
        width, channels = spec["d_model"], spec["bev_channels"]
        cells = math.prod(spec["latent_shape"])
        self.bev_downscale = nn.Conv2d(spec["latent_channels"], width, 1)
        self.keyval_embedding = nn.Embedding(cells + 1, width)
        self.query_embedding = nn.Embedding(spec["num_agents"] + 1, width)
        decoder_layer = nn.TransformerDecoderLayer(
            width, spec["num_heads"], spec["dim_feedforward"], spec["scene_dropout"],
            batch_first=True,
        )
        self.scene_decoder = nn.TransformerDecoder(decoder_layer, spec["scene_num_layers"])
        self.c5_conv = nn.Conv2d(spec["latent_channels"], channels, 1)
        self.up_conv5 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up_conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bev_proj = nn.Sequential(nn.Linear(width + channels, width), nn.ReLU(), nn.LayerNorm(width))
        self.plan_anchor_encoder = nn.Sequential(
            nn.Linear(spec["num_poses"] * 64, width), nn.ReLU(), nn.LayerNorm(width),
            nn.Linear(width, width),
        )
        self.cross_bev_attention = _GridSampleCrossBEVAttention(spec)
        self.cross_agent_attention = nn.MultiheadAttention(
            width, spec["num_heads"], dropout=spec["scene_dropout"], batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            width, spec["num_heads"], dropout=spec["scene_dropout"], batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(width, spec["dim_feedforward"]), nn.ReLU(),
                                 nn.Linear(spec["dim_feedforward"], width))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The outer score model is trainable; no dropout or statistics in the
        # pretrained feature extraction path may change with that mode.
        return super().train(False)

    @torch.no_grad()
    def prepare_scene(self, latent, status):
        expected = (self.spec["latent_channels"], *self.spec["latent_shape"])
        if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"Expected compact DD backbone [B,{expected}], got {tuple(latent.shape)}")
        if status.shape != (latent.shape[0], self.spec["d_model"]):
            raise ValueError("Expected DD encoded status [B,256]")
        # Cache is FP16, but frozen processing uses the same FP32 arithmetic in
        # training, validation and inference, independent of outer score AMP.
        with torch.autocast(device_type=latent.device.type, enabled=False):
            latent, status = latent.float(), status.float()
            low = self.bev_downscale(latent).flatten(-2, -1).permute(0, 2, 1)
            memory = torch.cat((low, status[:, None]), dim=1)
            memory = memory + self.keyval_embedding.weight[None]
            context = memory[:, :-1]
            query = self.query_embedding.weight[None].repeat(len(latent), 1, 1)
            query = self.scene_decoder(query, memory)
            ego, agents = query[:, :1], query[:, 1:]
            p5 = F.relu(self.c5_conv(latent))
            p4 = F.relu(self.up_conv5(F.interpolate(
                p5, scale_factor=self.spec["bev_upsample_factor"], mode="bilinear", align_corners=False,
            )))
            high = F.relu(self.up_conv4(F.interpolate(
                p4, size=self.spec["bev_shape"], mode="bilinear", align_corners=False,
            )))
            low = context.permute(0, 2, 1).contiguous().view(len(latent), -1, *self.spec["latent_shape"])
            low = F.interpolate(low, size=high.shape[2:], mode="bilinear", align_corners=False)
            bev = torch.cat((low, high), dim=1)
            bev = self.bev_proj(bev.flatten(-2, -1).permute(0, 2, 1))
            bev = bev.permute(0, 2, 1).contiguous().view(len(latent), -1, *self.spec["bev_shape"])
            value = self.cross_bev_attention.value_proj(bev)
            return {"context": context, "status": status, "ego_query": ego,
                    "agent_queries": agents, "bev_value": value}

    @torch.no_grad()
    def read_trajectory(self, physical_xy, scene):
        if physical_xy.ndim != 3 or tuple(physical_xy.shape[1:]) != (8, 2):
            raise ValueError("Expected current physical XY trajectory [B,8,2]")
        if len(physical_xy) != len(scene["context"]):
            raise ValueError("Trajectory and scene batch sizes differ")
        with torch.autocast(device_type=physical_xy.device.type, enabled=False):
            xy = physical_xy.float()[:, None]
            query = self.plan_anchor_encoder(_trajectory_position_embedding(xy).flatten(-2))
            query = self.cross_bev_attention(query, xy, scene["bev_value"])
            agents, ego = scene["agent_queries"], scene["ego_query"]
            query = self.norm1(query + self.dropout(self.cross_agent_attention(query, agents, agents)[0]))
            query = self.norm2(query + self.dropout1(self.cross_ego_attention(query, ego, ego)[0]))
            # Pinned DD uses norm3(ffn(query)), with NO FFN residual.
            return self.norm3(self.ffn(query))


class DDReaderScore(nn.Module):
    """Trainable conditional score with fixed DD semantics and a moving reader."""

    def __init__(self, model_config, normalizer, reader_spec):
        super().__init__()
        settings = dict(model_config)
        if settings.pop("architecture", None) != ARCHITECTURE:
            raise ValueError(f"Expected model.architecture={ARCHITECTURE}")
        if settings.get("parameterization", "score") != "score":
            raise ValueError("Frozen no-grad reader currently supports direct score only")
        if (settings.get("future_len"), settings.get("input_dim"), settings.get("context_dim")) != (8, 4, 256):
            raise ValueError("DD reader score requires eight poses, four channels, width-256 conditions")
        self.reader_spec = copy.deepcopy(reader_spec)
        self.reader = FrozenDDReader(reader_spec)
        self.score = ScoreFunctionBranch(**settings)
        mean, std = (torch.as_tensor(normalizer[k], dtype=torch.float32) for k in ("mean", "std"))
        if (mean.shape != (4,) or std.shape != (4,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()
                or (mean[2:] != 0).any() or (std[2:] != 1).any()):
            raise ValueError("Invalid four-channel NAVSIM training normalizer")
        self.register_buffer("normalizer_mean", mean.clone())
        self.register_buffer("normalizer_std", std.clone())
        self.parameterization = "score"
        self.uses_neighbor_future = False

    def prepare_scene(self, latent, status):
        return self.reader.prepare_scene(latent, status)

    def score_prepared(self, noisy_norm, scene):
        if noisy_norm.ndim != 3 or tuple(noisy_norm.shape[1:]) != (8, 4):
            raise ValueError("Expected noisy/current normalized trajectory [B,8,4]")
        with torch.no_grad(), torch.autocast(device_type=noisy_norm.device.type, enabled=False):
            physical_xy = (noisy_norm.float() * self.normalizer_std + self.normalizer_mean)[..., :2]
            trajectory_context = self.reader.read_trajectory(physical_xy, scene)
            # 64 BEV + 1 ego + 30 agents + 1 trajectory-conditioned token.
            context = torch.cat((scene["context"], scene["ego_query"],
                                 scene["agent_queries"], trajectory_context), dim=1)
        return self.score(noisy_norm, context, scene["status"])

    def forward(self, noisy_norm, latent, status, **kwargs):
        if kwargs:
            raise ValueError("DD reader score does not accept neighbor futures")
        return self.score_prepared(noisy_norm, self.prepare_scene(latent, status))


def export_reader_assets(agent):
    """Extract only used pretrained tensors from a strictly loaded native agent."""
    model = agent._transfuser_model
    config, backbone = model._config, model._backbone
    layer = model._trajectory_head.diff_decoder.layers[0]
    spec = {
        "latent_channels": model._bev_downscale.in_channels,
        "latent_shape": [8, 8], "d_model": config.tf_d_model,
        "bev_channels": config.bev_features_channels,
        "bev_shape": [config.lidar_resolution_height // config.bev_down_sample_factor,
                      config.lidar_resolution_width // config.bev_down_sample_factor],
        "bev_upsample_factor": config.bev_upsample_factor,
        "num_agents": config.num_bounding_boxes, "scene_num_layers": config.tf_num_layers,
        "num_heads": config.tf_num_head, "dim_feedforward": config.tf_d_ffn,
        "scene_dropout": config.tf_dropout, "lidar_max_x": config.lidar_max_x,
        "lidar_max_y": config.lidar_max_y, "num_poses": 8,
    }
    modules = {
        "bev_downscale": model._bev_downscale, "keyval_embedding": model._keyval_embedding,
        "query_embedding": model._query_embedding, "scene_decoder": model._tf_decoder,
        "c5_conv": backbone.c5_conv, "up_conv5": backbone.up_conv5, "up_conv4": backbone.up_conv4,
        "bev_proj": model.bev_proj, "plan_anchor_encoder": model._trajectory_head.plan_anchor_encoder,
        "cross_bev_attention": layer.cross_bev_attention,
        "cross_agent_attention": layer.cross_agent_attention, "cross_ego_attention": layer.cross_ego_attention,
        "norm1": layer.norm1, "norm2": layer.norm2, "norm3": layer.norm3, "ffn": layer.ffn,
    }
    state = {f"{prefix}.{key}": value.detach().cpu().clone()
             for prefix, module in modules.items() for key, value in module.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        check = FrozenDDReader(spec)
    check.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite pretrained DD reader tensor")
    return {"format": ASSET_FORMAT, "dd_commit": DD_COMMIT, "spec": spec, "state_dict": state}


def build_reader_score(model_config, normalizer, assets, device):
    if assets.get("format") != ASSET_FORMAT or assets.get("dd_commit") != DD_COMMIT:
        raise ValueError("Unsupported or unpinned DD reader assets")
    model = DDReaderScore(model_config, normalizer, assets["spec"])
    model.reader.load_state_dict(assets["state_dict"], strict=True)
    return model.to(device)


@torch.no_grad()
def verify_native_reader(agent, latent, status, physical_xy, assets=None, *, rtol=1e-5, atol=1e-5):
    """Compare scene representation and the full first-layer prefix to native DD.

    Both paths receive the SAME latent/status values. Call with original FP32
    values to test source parity, and optionally FP16 round-tripped values to
    exercise the cache contract. No native sampler is invoked.
    """
    model = agent.eval().requires_grad_(False)._transfuser_model
    assets = export_reader_assets(agent) if assets is None else assets
    with torch.random.fork_rng(devices=[]):
        reader = FrozenDDReader(assets["spec"]).to(latent.device)
    reader.load_state_dict(assets["state_dict"], strict=True)
    with torch.autocast(device_type=latent.device.type, enabled=False):
        latent, status, physical_xy = latent.float(), status.float(), physical_xy.float()
        batch = len(latent)
        low = model._bev_downscale(latent).flatten(-2, -1).permute(0, 2, 1)
        keyval = torch.cat((low, status[:, None]), dim=1)
        keyval = keyval + model._keyval_embedding.weight[None]
        query = model._query_embedding.weight[None].repeat(batch, 1, 1)
        query = model._tf_decoder(query, keyval)
        ego, agents = query.split(model._query_splits, dim=1)
        high = model._backbone.top_down(latent)
        cross = keyval[:, :-1].permute(0, 2, 1).contiguous().view(batch, -1, *latent.shape[-2:])
        cross = F.interpolate(cross, size=high.shape[-2:], mode="bilinear", align_corners=False)
        cross = torch.cat((cross, high), dim=1)
        cross = model.bev_proj(cross.flatten(-2, -1).permute(0, 2, 1))
        cross = cross.permute(0, 2, 1).contiguous().view(batch, -1, *high.shape[-2:])
        # Import the original implementation for an independent positional and
        # spatial-lookup comparison on a server with pinned DD installed.
        from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
        xy = physical_xy[:, None]
        trajectory = model._trajectory_head.plan_anchor_encoder(
            gen_sineembed_for_position(xy, hidden_dim=64).flatten(-2)
        )
        layer = model._trajectory_head.diff_decoder.layers[0]
        trajectory = layer.cross_bev_attention(trajectory, xy, cross, high.shape[-2:])
        trajectory = layer.norm1(trajectory + layer.dropout(layer.cross_agent_attention(trajectory, agents, agents)[0]))
        trajectory = layer.norm2(trajectory + layer.dropout1(layer.cross_ego_attention(trajectory, ego, ego)[0]))
        trajectory = layer.norm3(layer.ffn(trajectory))
        scene = reader.prepare_scene(latent, status)
        actual = reader.read_trajectory(physical_xy, scene)
    pairs = {"context": (scene["context"], keyval[:, :-1]), "ego_query": (scene["ego_query"], ego),
             "agent_queries": (scene["agent_queries"], agents), "reader": (actual, trajectory)}
    errors = {}
    for name, (observed, expected) in pairs.items():
        torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol)
        errors[name + "_max_abs_error"] = (observed - expected).abs().max().item()
    return errors
