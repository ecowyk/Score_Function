import json
from pathlib import Path
import tempfile
import unittest

import torch

from score_function.navsim_chainflow.runtime import CHECKPOINT_FORMAT
from score_function.navsim_chainflow.sweep_checkpoints import prepare_checkpoints
from score_function.navsim_chainflow.train import TRAIN_FORMAT
from score_function.utils.train_utils import fingerprint, file_hash, load_tensor


class SweepCheckpoints(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = dict(epochs=40, global_batch_size=64, learning_rate=.0002, weight_decay=.01,
                         model={"dropout": .1}, gamma=.1, sigma=.1, precision="32-true", num_workers=2)
        self.reference = self.root / "reference"
        self.sweep = self.root / "sweep"
        self.output = self.root / "eval"
        self.plan = dict(paths={"training": str(self.reference)}, config=self.base)
        for group, final_epoch in [("reference", 40), ("wd", 40), ("dropout", 40), ("lr", 39), ("lr_wd_dropout", 39)]:
            parts = group.split("_")
            directory = self.reference if group == "reference" else self.sweep / group / "training"
            directory.mkdir(parents=True)
            config = dict(self.base, model={"dropout": .2 if "dropout" in parts else .1},
                          learning_rate=.0001 if "lr" in parts else .0002, weight_decay=.05 if "wd" in parts else .01,
                          num_workers=4 if group == "lr" else 2)
            protocol = dict(config={k: v for k, v in config.items() if k != "num_workers"}, cache_fingerprint="shared-cache")
            signature = fingerprint(protocol)
            (directory / "run.json").write_text(json.dumps(dict(config=config, protocol=protocol, signature=signature,
                                                                updates_per_epoch=1329)))
            common = dict(format=CHECKPOINT_FORMAT, config=config, model=config["model"], sigma=.1, gamma=.1,
                          training_precision=config["precision"], normalizer={"std": [2, 2, 1, 1]},
                          conditioner_identity={"source": "chainflow"}, training_signature=signature,
                          cache_fingerprint="shared-cache", weight_kind="ema")
            history = [dict(epoch=e, val_dsm_ema=abs(18-e) + .1) for e in range(1, final_epoch + 1)]
            best = dict(state_dict={"w": torch.tensor([18.])}, epoch=18, val_dsm=.1)
            ema = {"w": torch.tensor([float(final_epoch)])}
            last = dict(format=TRAIN_FORMAT, signature=signature, history=history, step=final_epoch*1329,
                        ema=ema, raw={"w": torch.tensor([-1.])}, best={"ema": best})
            torch.save(last, directory / "last.pt")
            torch.save(dict(common, **best), directory / "best_ema.pt")
            if final_epoch == 40:
                torch.save(dict(common, state_dict=ema, epoch=40, val_dsm=history[-1]["val_dsm_ema"]),
                           directory / "final_ema.pt")
                (directory / "TRAINING_COMPLETE.json").write_text(json.dumps(dict(epochs=40, signature=signature,
                                                    final_ema_sha256=file_hash(directory / "final_ema.pt"))))

    def prepare(self):
        return prepare_checkpoints(self.plan, self.sweep, self.output)

    def test_completed_and_interrupted_groups_select_correct_ema_and_repeat(self):
        records = self.prepare()
        self.assertEqual(len(records), 10)
        self.assertEqual(records["reference/final"]["sha256"], file_hash(self.reference / "final_ema.pt"))
        for label, selection, epoch in [("lr/final", "last_available", 39), ("lr_wd_dropout/final", "last_available", 39),
                                        ("wd/final", "final", 40), ("dropout/best", "best", 18)]:
            record = records[label]
            self.assertEqual(record["selection"], selection)
            port = load_tensor(record["path"])
            self.assertEqual(port["weight_kind"], "ema")
            self.assertEqual(port["epoch"], epoch)
            self.assertEqual(port["state_dict"]["w"].item(), epoch)
        self.assertFalse((self.sweep / "lr/training/final_ema.pt").exists())
        self.assertEqual(records, self.prepare())

    def test_refuses_swapped_best_state_from_another_epoch(self):
        path = self.sweep / "wd/training/best_ema.pt"
        checkpoint = load_tensor(path)
        checkpoint["state_dict"]["w"] = torch.tensor([42.])
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "weights disagree"):
            self.prepare()

    def test_recovers_interruption_between_full_checkpoint_and_best_export(self):
        path = self.sweep / "lr/training/best_ema.pt"
        checkpoint = load_tensor(path)
        checkpoint.update(epoch=17, val_dsm=1.1, state_dict={"w": torch.tensor([17.])})
        torch.save(checkpoint, path)
        record = self.prepare()["lr/best"]
        self.assertEqual(load_tensor(record["path"])["state_dict"]["w"].item(), 18)
        self.assertEqual(load_tensor(path)["epoch"], 17)
