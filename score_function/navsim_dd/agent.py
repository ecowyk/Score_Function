"""NAVSIM AbstractAgent for running a trained score via the upstream evaluator."""

from __future__ import annotations

import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory

from score_function.navsim_dd.cache import conditioner_identity, verified_devkit_root
from score_function.navsim_dd.conditioner import DDConditioner, features_from_input, load_diffusiondrive
from score_function.navsim_dd.runtime import NavsimRefiner


class DiffusionDriveScoreAgent(AbstractAgent):
    def __init__(self, checkpoint, anchor, backbone, score_checkpoint=None, steps=20,
                 gamma=None, device="cpu"):
        super().__init__(requires_scene=False)
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if steps and not score_checkpoint:
            raise ValueError("Pass a trained NAVSIM score checkpoint for K > 0")
        self.assets = (checkpoint, anchor, backbone)
        self.score_checkpoint = score_checkpoint
        self.steps = steps
        self.gamma = gamma
        self.device = torch.device(device)
        self.native = None
        self.refiner = None
        self.conditioner = None

    def name(self):
        return f"DiffusionDriveScoreK{self.steps}"

    def get_sensor_config(self):
        return SensorConfig.build_all_sensors(include=[3])

    def initialize(self):
        verified_devkit_root()
        self.native = load_diffusiondrive(*self.assets, device=self.device)
        if self.steps:
            self.refiner = NavsimRefiner.from_checkpoint(self.score_checkpoint, self.device)
            self.refiner.require_conditioner(conditioner_identity(*self.assets))
            self.conditioner = DDConditioner(self.native)
        self.eval()

    @torch.no_grad()
    def compute_trajectory(self, agent_input):
        if self.native is None:
            raise RuntimeError("initialize() must run before planning")
        features = features_from_input(self.native, agent_input, self.device)
        raw = self.native(features)["trajectory"]
        if self.steps:
            scene, status = self.conditioner.encode_features(features)
            raw = self.refiner.refine(raw, scene, status, self.steps, gamma=self.gamma)
        return Trajectory(raw[0].cpu().numpy(), self.native._config.trajectory_sampling)
