"""Official paired state perturbation before frozen condition encoding.

The clean tensor cache still defines the split and validation targets. Training
loads its original NPZ records anew, so every visit can receive a fresh official
augmentation. This module deliberately does not alter the score architecture.
"""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from score_function.utils.feature_dataset import INPUT_KEYS
from score_function.utils.neighbor import prediction_neighbors
from score_function.utils.normalizer import normalize_ego_future
from score_function.utils.planner_utils import encode_conditions, load_frozen_planner
from score_function.utils.train_utils import file_hash, fingerprint, read_json, resolve_path

RAW_INPUT_KEYS = INPUT_KEYS + ("route_lanes_speed_limit", "route_lanes_has_speed_limit")
RAW_KEYS = RAW_INPUT_KEYS + ("ego_agent_future", "neighbor_agents_future")
RECORD_IDENTITY_KEYS = ("token", "recording", "split", "start_time_us")


class RawTrainingDataset(Dataset):
    """Resolve raw features using the exact records of an existing clean cache.

    Each NPZ is read once into memory, hashed against the already published
    digest, and decoded from those same bytes. There is no full-corpus startup
    hash pass, extra disk read for checksums, or mutable per-worker audit cache.
    This also detects changed inputs on later epochs and after resume.
    """

    def __init__(self, cached_dataset, config):
        self.cached_dataset = cached_dataset
        self.metadata = dict(cached_dataset.metadata)
        self.root = Path(config["paths"]["root"])
        self.records = list(cached_dataset.records)
        fallback_hash = None
        if any(not row.get("feature") or not row.get("feature_sha256") for row in self.records):
            manifest_path = resolve_path(config, "manifest")
            manifest = read_json(manifest_path)
            by_token = {row["token"]: row for row in manifest}
            if len(by_token) != len(manifest):
                raise ValueError("Duplicate tokens in raw feature manifest")
            resolved = []
            for record in self.records:
                raw = by_token.get(record["token"])
                if raw is None or any(record[key] != raw.get(key) for key in RECORD_IDENTITY_KEYS):
                    raise ValueError("Raw manifest does not match cached frame/split identity")
                for key in ("feature", "feature_sha256"):
                    if record.get(key) and record[key] != raw.get(key):
                        raise ValueError(f"Raw manifest conflicts with cached {key}")
                resolved.append(
                    {**record, "feature": raw["feature"], "feature_sha256": raw["feature_sha256"]}
                )
            self.records = resolved
            fallback_hash = file_hash(manifest_path)
        for record in self.records:
            digest = record.get("feature_sha256", "")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Raw features require published SHA-256 digests")
        self.sha256 = fingerprint(
            {
                "clean_cache": cached_dataset.sha256,
                "raw_manifest_fallback": fallback_hash,
                "protocol": "official_state_perturbation_before_normalization_v1",
                "implementation": file_hash(__file__),
                "augmentation": config["training"]["data_augmentation"],
            }
        )
        self.metadata["training_condition_source"] = (
            "online official StatePerturbation, frozen scene/route encoders; "
            "neighbor variants predict fresh neighbors from the same augmented inputs"
        )
        self.metadata["raw_feature_verification"] = "SHA-256 of every NPZ on every read"

    def __len__(self):
        return len(self.records)

    def close(self):
        self.cached_dataset.close()

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["feature"]).expanduser()
        if not path.is_absolute():
            path = self.root / path
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != record["feature_sha256"]:
            raise ValueError(f"Changed raw feature NPZ: {path}")
        with np.load(BytesIO(contents), allow_pickle=False) as source:
            missing = set(RAW_KEYS).difference(source.files)
            if missing:
                raise ValueError(f"Official raw NPZ is missing {sorted(missing)}: {path}")
            row = {}
            for key in RAW_KEYS:
                value = torch.from_numpy(np.array(source[key], copy=True))
                if value.is_floating_point():
                    value = value.float()
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite raw {key}: {path}")
                row[f"raw_{key}"] = value
        row["record"] = record
        return row


class OnlineTrainingProvider:
    """Convert one collated raw batch into DSM inputs without planner gradients.

    Encoder and (when requested) DP sampler calls are chunked independently of
    the score microbatch, bounding their activation memory. Augmentation is
    called on every training visit and consumes the checkpointed training RNG.
    Never use this provider for validation or substitute clean neighbor caches.
    """

    def __init__(self, config, device):
        from diffusion_planner.utils.data_augmentation import StatePerturbation

        settings = config["training"]["data_augmentation"]
        self.device = torch.device(device)
        self.batch_size = settings["encoding_batch_size"]
        self.use_neighbors = config["model"].get("neighbor_future", False)
        self.planner, self.args = load_frozen_planner(config, self.device)
        self.planner.eval().requires_grad_(False)
        # Upstream applies perturbations when rand >= augment_prob (and |vx|>=2).
        # Expose an actual application probability; 0.5 is identical to DP.
        self.augmentation = StatePerturbation(
            augment_prob=1.0 - settings["probability"], device=self.device
        )

    @torch.no_grad()
    def __call__(self, batch):
        self.planner.eval()
        size = batch["raw_ego_agent_future"].shape[0]
        chunks = []
        for start in range(0, size, self.batch_size):
            stop = min(start + self.batch_size, size)
            # The upstream augmentor mutates tensors. Own each chunk so reusable
            # raw batches and loader memory never acquire augmented coordinates.
            raw = {
                key: batch[f"raw_{key}"][start:stop].to(self.device).clone()
                for key in RAW_INPUT_KEYS
            }
            raw["neighbor_agents_past"] = raw["neighbor_agents_past"][:, : self.args.agent_num]
            future = batch["raw_ego_agent_future"][start:stop].to(self.device).clone()
            neighbors = (
                batch["raw_neighbor_agents_future"][start:stop, : self.args.predicted_neighbor_num]
                .to(self.device)
                .clone()
            )
            if future.shape[1:] != (80, 3) or neighbors.shape[1:] != (10, 80, 3):
                raise ValueError("Require official raw ego [B,80,3] and neighbors [B,10,80,3]")
            # Match official train_epoch ordering exactly: paired augmentation,
            # future heading packing, observation normalization, then encoding.
            raw, future, _ = self.augmentation(raw, future, neighbors)
            target = torch.cat(
                (future[..., :2], future[..., 2:3].cos(), future[..., 2:3].sin()), dim=-1
            )
            inputs = self.args.observation_normalizer(raw)
            context, route = encode_conditions(self.planner, inputs)
            result = {
                "target": normalize_ego_future(target, self.args.state_normalizer),
                "context": context,
                "route": route,
            }
            if self.use_neighbors:
                # Use this very same augmented observation and encoding. Do not
                # overwrite its ego state with the clean online-adapter placeholder.
                output = self.planner.decoder({"encoding": context}, inputs)
                result.update(
                    prediction_neighbors(output["prediction"], inputs, self.args.state_normalizer)
                )
            if any(not torch.isfinite(value).all() for value in result.values()):
                raise FloatingPointError("Nonfinite online training features")
            chunks.append(result)
        if not chunks:
            raise ValueError("Empty online training batch")
        return {
            **{key: torch.cat([chunk[key] for chunk in chunks]).detach() for key in chunks[0]},
            "tokens": batch["tokens"],
        }
