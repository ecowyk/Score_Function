"""Isolate added Score work from the native planner's random streams."""
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_random_state(device="cpu"):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
