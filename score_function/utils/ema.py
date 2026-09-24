"""Exponential moving average used by the retained trainer."""

import torch


@torch.no_grad()
def update_ema(averaged, model, decay):
    for target, source in zip(averaged.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1.0 - decay)
    for target, source in zip(averaged.buffers(), model.buffers()):
        target.copy_(source)
