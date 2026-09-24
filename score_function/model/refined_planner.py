"""Compose the untouched frozen generator with the separate clean-space score branch."""

import torch
from torch import nn

from score_function.model.refinement import refine_ego
from score_function.utils.neighbor import prediction_neighbors
from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future


class ScoreRefinedPlanner(nn.Module):
    def __init__(
        self, base_model, score_branch, normalizer, sigma, gamma, steps, heading_projection=True
    ):
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.score_branch = score_branch
        self.normalizer = normalizer
        self.sigma, self.gamma, self.steps = sigma, gamma, steps
        self.heading_projection = heading_projection
        self.last_refinement_trace = None
        self.record_trace = False
        self.last_baseline_ego = None
        self.last_refined_ego = None

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    @torch.no_grad()
    def refine_prediction(self, inputs, encoding, output, record_trace=False):
        if self.gamma == 0 or self.steps == 0:
            self.last_refinement_trace = {"disabled": True, "completed_steps": 0}
            return output
        prediction = output["prediction"]
        if prediction.ndim != 4 or prediction.shape[2:] != (80, 4):
            raise ValueError("Expected the official joint future prediction [B,P,80,4]")
        ego = normalize_ego_future(prediction[:, 0], self.normalizer)
        route = self.base_model.decoder.decoder.dit.route_encoder(inputs["route_lanes"])
        result, trace = refine_ego(
            self.score_branch,
            ego,
            encoding["encoding"],
            route,
            self.normalizer,
            self.sigma,
            self.gamma,
            self.steps,
            self.heading_projection,
            record_trace,
            **(
                prediction_neighbors(prediction, inputs, self.normalizer)
                if getattr(self.score_branch, "uses_neighbor_future", False)
                else {}
            ),
        )
        revised = prediction.clone()
        revised[:, 0] = denormalize_ego_future(result, self.normalizer)
        self.last_refinement_trace = trace
        return {**output, "prediction": revised}

    @torch.no_grad()
    def forward(self, inputs, record_trace=None):
        self.base_model.eval()
        encoding, output = self.base_model(inputs)
        record_trace = self.record_trace if record_trace is None else record_trace
        revised = self.refine_prediction(inputs, encoding, output, record_trace)
        self.last_baseline_ego = (
            output["prediction"][:, 0].detach().cpu().clone() if record_trace else None
        )
        self.last_refined_ego = (
            revised["prediction"][:, 0].detach().cpu().clone() if record_trace else None
        )
        return encoding, revised
