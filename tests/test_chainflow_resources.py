"""GPU leases must respect allocations and never infer idle compute from VRAM."""
import fcntl
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from score_function.navsim_chainflow import resources as r


def gpu(index, used=0, utilization=0):
    return r.GPU(index, f"GPU-test-{index}", 81920, used, utilization)


class ResourceTests(unittest.TestCase):
    def test_parse_and_invalid_telemetry(self):
        rows = r.parse_gpu_query("0, GPU-aaaa, 81920, 54553, 100\n")
        self.assertEqual(rows[0].free_mib, 27367)
        for bad in ("0,GPU-a,81920,0,N/A", "0,GPU-a,81920,81921,0",
                    "0,GPU-a,81920,0,0\n1,GPU-a,81920,0,0", "0,GPU-a,81920"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                r.parse_gpu_query(bad)

    def test_busy_with_free_memory_is_not_idle(self):
        busy = gpu(4, used=54553, utilization=100)
        self.assertFalse(r.eligible(busy, budget_gb=12, reserve_gb=4, policy="wait-idle"))
        self.assertTrue(r.eligible(busy, budget_gb=12, reserve_gb=4, policy="shared"))
        self.assertFalse(r.eligible(gpu(0, 73360, 99), budget_gb=12, reserve_gb=4, policy="shared"))
        self.assertFalse(r.eligible(gpu(0, 2000, 0), budget_gb=12, reserve_gb=4, policy="wait-idle"))

    def test_mask_intersection_and_invalid_requests(self):
        gpus = [gpu(i) for i in range(8)]
        self.assertEqual([x.index for x in r.select_pool(gpus, environ={"CUDA_VISIBLE_DEVICES": "GPU-test-4,GPU-test-6"})], [4, 6])
        self.assertEqual([x.index for x in r.select_pool(gpus, "6", environ={"CUDA_VISIBLE_DEVICES": "4,6"})], [6])
        self.assertEqual(r.select_pool(gpus, environ={"CUDA_VISIBLE_DEVICES": ""}), [])
        for selector in ("0", "4,4", "MIG-whatever", "GPU-test-", "9"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                r.select_pool(gpus, selector, environ={"CUDA_VISIBLE_DEVICES": "4,6"})

    def test_lease_lasts_for_body_and_releases_on_exception(self):
        with TemporaryDirectory() as tmp, patch.object(r, "query_gpus", return_value=[gpu(4)]):
            with self.assertRaisesRegex(RuntimeError, "body"):
                with r.acquire_gpus(workers=1, wait=False, lock_dir=tmp, environ={}) as selected:
                    self.assertEqual(r.gpu_environment(selected, 12)["CUDA_VISIBLE_DEVICES"], "GPU-test-4")
                    with (Path(tmp) / "GPU-test-4.lock").open("a+") as other:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("body")
            with r.acquire_gpus(workers=1, wait=False, lock_dir=tmp, environ={}) as selected:
                self.assertEqual(selected[0].index, 4)

    def test_insufficient_group_releases_partial_leases(self):
        with TemporaryDirectory() as tmp:
            with (Path(tmp) / "GPU-test-1.lock").open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(r, "query_gpus", return_value=[gpu(0), gpu(1)]):
                    with self.assertRaises(TimeoutError):
                        with r.acquire_gpus(workers=2, wait=False, lock_dir=tmp, environ={}):
                            self.fail("Should not acquire partially available group")
                    with (Path(tmp) / "GPU-test-0.lock").open("a+") as released:
                        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_wait_then_acquire_and_callback(self):
        statuses = []
        with TemporaryDirectory() as tmp, patch.object(r, "time") as clock, patch.object(
            r, "query_gpus", side_effect=[[gpu(0, 54553, 100)], [gpu(0)], [gpu(0)]]
        ):
            clock.monotonic.return_value = 0
            with r.acquire_gpus(workers=1, lock_dir=tmp, on_status=statuses.append, environ={}):
                pass
            clock.sleep.assert_called_once_with(15)
        self.assertEqual([s["state"] for s in statuses], ["waiting", "acquired"])

    def test_post_lease_refresh_can_prevent_launch(self):
        with TemporaryDirectory() as tmp, patch.object(
            r, "query_gpus", side_effect=[[gpu(0)], [gpu(0, 60000, 100)]]
        ):
            with self.assertRaises(TimeoutError):
                with r.acquire_gpus(workers=1, wait=False, lock_dir=tmp, environ={}):
                    self.fail("GPU became busy before launch")

    def test_auto_group_and_shared_notice(self):
        statuses = []
        gpus = [gpu(i, 54553, 100) for i in range(5)]
        with TemporaryDirectory() as tmp, patch.object(r, "query_gpus", return_value=gpus):
            with r.acquire_gpus(workers="auto", policy="shared", wait=False, lock_dir=tmp,
                                on_status=statuses.append, environ={}) as selected:
                self.assertEqual(len(selected), 4)
                with (Path(tmp) / "GPU-test-4.lock").open("a+") as unused:
                    fcntl.flock(unused, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertIn("may slow", statuses[0]["notice"])

    def test_cancellation_leaves_no_partial_lease(self):
        with TemporaryDirectory() as tmp, patch.object(r, "query_gpus", return_value=[gpu(0), gpu(1, 60000, 100)]), \
                patch.object(r.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                with r.acquire_gpus(workers=2, lock_dir=tmp, environ={}):
                    self.fail("Only one GPU idle")
            with (Path(tmp) / "GPU-test-0.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_callback_failure_releases_acquired_lease(self):
        with TemporaryDirectory() as tmp, patch.object(r, "query_gpus", return_value=[gpu(0)]):
            def fail(_):
                raise RuntimeError("status write failed")
            with self.assertRaisesRegex(RuntimeError, "status write"):
                with r.acquire_gpus(workers=1, wait=False, lock_dir=tmp, on_status=fail, environ={}):
                    self.fail("Callback failed")
            with (Path(tmp) / "GPU-test-0.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
