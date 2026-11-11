"""Download corruption/range and archive safety checks without network or GPUs."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from score_function.navsim_chainflow import assets


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}


def asset_for(body, path="weights/example.ckpt", role="checkpoint"):
    return {"repo": "example/release", "repo_type": "model", "revision": "a" * 40,
            "path": path, "role": role, "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "checkpoint.bin"

    def tearDown(self):
        self.temporary.cleanup()

    def download(self, body, responses, partial=None, attempts=None):
        if partial is not None:
            self.destination.with_suffix(".bin.part").write_bytes(partial)
        opener = mock.Mock(side_effect=responses)
        result = assets.download(asset_for(body), self.destination, opener=opener,
                                 pause=lambda _: None, attempts=attempts or len(responses))
        self.assertEqual(result.read_bytes(), body)
        return opener

    def test_valid_range_resume(self):
        body = b"binary asset bytes"
        opener = self.download(body, [Response(body[6:], 206, {
            "Content-Range": f"bytes 6-{len(body)-1}/{len(body)}",
            "Content-Length": str(len(body)-6)})], partial=body[:6])
        self.assertEqual(opener.call_args.args[0].get_header("Range"), "bytes=6-")

    def test_ignored_range_preserves_partial_without_reading_response(self):
        body = b"binary asset bytes"
        partial = self.destination.with_suffix(".bin.part")
        partial.write_bytes(body[:6])
        responses = [Response(body), Response(body)]
        for response in responses:
            response.read = mock.Mock(side_effect=AssertionError("Must not read rejected body"))
        opener = mock.Mock(side_effect=responses)
        with self.assertRaises(RuntimeError):
            assets.download(asset_for(body), self.destination, opener=opener,
                            pause=lambda _: None, attempts=2)
        self.assertEqual(partial.read_bytes(), body[:6])
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob("*.bad-*")))
        for response in responses:
            response.read.assert_not_called()
        self.assertEqual([call.args[0].get_header("Range") for call in opener.call_args_list],
                         ["bytes=6-", "bytes=6-"])

    def test_ignored_range_then_valid_response_resumes_same_prefix(self):
        body = b"binary asset bytes"
        ignored = Response(body)
        ignored.read = mock.Mock(side_effect=AssertionError("Must not read rejected body"))
        opener = self.download(body, [ignored, Response(body[6:], 206, {
            "Content-Range": f"bytes 6-{len(body)-1}/{len(body)}",
            "Content-Length": str(len(body)-6)})], partial=body[:6])
        ignored.read.assert_not_called()
        self.assertEqual([call.args[0].get_header("Range") for call in opener.call_args_list],
                         ["bytes=6-", "bytes=6-"])

    def test_mismatched_range_never_appends(self):
        partial = self.destination.with_suffix(".bin.part")
        partial.write_bytes(b"123")
        response = Response(b"456789", 206, {"Content-Range": "bytes 2-7/9"})
        with self.assertRaises(RuntimeError):
            assets.download(asset_for(b"123456789"), self.destination,
                            opener=lambda _: response, attempts=1)
        self.assertEqual(partial.read_bytes(), b"123")
        self.assertFalse(self.destination.exists())

    def test_corrupt_prefix_quarantined_then_fresh_retry(self):
        body = b"binary asset bytes"
        responses = [Response(body[3:], 206, {"Content-Range": f"bytes 3-{len(body)-1}/{len(body)}"}), Response(body)]
        opener = self.download(body, responses, partial=b"BAD")
        self.assertEqual(opener.call_args_list[0].args[0].get_header("Range"), "bytes=3-")
        self.assertIsNone(opener.call_args_list[1].args[0].get_header("Range"))
        self.assertEqual(len(list(self.root.glob("*.bad-*"))), 1)

    def test_short_response_resumes_at_actual_length(self):
        body = b"binary asset bytes"
        responses = [Response(body[:6], 200, {"Content-Length": str(len(body))}),
                     Response(body[6:], 206, {"Content-Range": f"bytes 6-{len(body)-1}/{len(body)}"})]
        opener = self.download(body, responses)
        self.assertEqual(opener.call_args_list[1].args[0].get_header("Range"), "bytes=6-")

    def test_error_document_cannot_be_published(self):
        for headers in ({"Content-Type": "text/html"}, {}):
            with self.subTest(headers=headers), self.assertRaises(RuntimeError):
                assets.download(asset_for(b"<html>proxy error</html>"), self.destination,
                                opener=lambda _: Response(b"<html>proxy error</html>", headers=headers), attempts=1)
            self.assertFalse(self.destination.exists())

    def test_oversized_existing_partial_quarantined(self):
        self.download(b"binary", [Response(b"binary")], partial=b"too long to be a real partial")
        self.assertEqual(len(list(self.root.glob("*.bad-*"))), 1)

    def test_completed_wrong_hash_is_not_reused(self):
        self.destination.write_bytes(b"wrong!")
        self.download(b"binary", [Response(b"binary")])
        self.assertEqual(len(list(self.root.glob("*.bad-*"))), 1)

    def test_release_identity_stable_across_native_then_full_inventory(self):
        plan = assets.load_plan()
        release = self.root / "release_manifest.json"
        assets.publish_release_manifest(release, plan)
        before = release.read_bytes()
        assets.atomic_json(self.root / "assets.json", {"native_complete": True, "complete": False})
        assets.publish_release_manifest(release, plan)
        assets.atomic_json(self.root / "assets.json", {"native_complete": True, "complete": True})
        assets.publish_release_manifest(release, plan)
        self.assertEqual(before, release.read_bytes())
        self.assertEqual(len(json.loads(before)["assets"]), 27)
        self.assertNotIn("local_path", before.decode())
        altered = dict(plan, training_split="wrong")
        with self.assertRaises(ValueError):
            assets.publish_release_manifest(release, altered)
        self.assertEqual(before, release.read_bytes())

    def test_official_pins_and_native_cost(self):
        plan = assets.load_plan()
        native = assets.selected_assets(plan, scope="native")
        self.assertEqual(len(native), 3)
        self.assertEqual(sum(row["size"] for row in native), 109598014883)
        self.assertEqual(sum(row["size"] for row in plan["assets"] if row["role"] == "vlm_archive"), 1035243638403)
        self.assertEqual(len(assets.selected_assets(plan, "trainval")), 24)

    def test_trainval_only_preparation_never_selects_existing_native_assets(self):
        plan = assets.load_plan()
        selected = assets.selected_assets(plan, only="trainval", scope="all")
        self.assertEqual(len(selected), 24)
        self.assertTrue(all(row["role"] == "vlm_archive" and row["path"].startswith("trainval/")
                            for row in selected))
        native_hashes = {row["sha256"] for row in assets.selected_assets(plan, scope="native")}
        self.assertFalse(native_hashes & {row["sha256"] for row in selected})


class ArchiveTests(unittest.TestCase):
    log = "2021.05.12.19.36.12_veh-35_00005_00204"
    token = "0a678d2136b35b56"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.relative = f"{self.log}/{self.token}/internvl_feature.gz"

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, members):
        import zstandard
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            for name, body, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(body)
                    tar.addfile(info, io.BytesIO(body))
                else:
                    info.linkname = "../../escape"
                    tar.addfile(info)
        payload = zstandard.ZstdCompressor().compress(data.getvalue())
        archive = self.root / "test.tar.zst"
        archive.write_bytes(payload)
        return archive, asset_for(payload, "navtest/test.tar.zst", "vlm_archive")

    def test_navtest_wrapper_unified_and_targets_not_extracted(self):
        prefix = assets.NAVTEST_PREFIX
        archive, asset = self.archive([
            (prefix, b"", tarfile.DIRTYPE),
            (f"{prefix}/{self.relative}", b"feature data", tarfile.REGTYPE),
            (f"{prefix}/{self.log}/{self.token}/trajectory_target.gz", b"target", tarfile.REGTYPE)])
        marker = assets.extract_archive(archive, self.cache, asset)
        target = self.cache / self.relative
        self.assertEqual(target.read_bytes(), b"feature data")
        self.assertFalse(target.with_name("trajectory_target.gz").exists())
        self.assertTrue(assets.extraction_ready(marker, self.cache, asset, full_hash=True))
        self.assertFalse((self.cache / prefix).exists())

    def test_trainval_bare_layout_and_existing_equal_file_preserved(self):
        archive, asset = self.archive([(self.relative, b"feature data", tarfile.REGTYPE)])
        target = self.cache / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b"feature data")
        before = target.stat().st_mtime_ns
        marker = assets.extract_archive(archive, self.cache, asset)
        self.assertEqual(before, target.stat().st_mtime_ns)
        self.assertTrue(assets.extraction_ready(marker, self.cache, asset))
        target.write_bytes(b"changed data")
        with self.assertRaises(ValueError):
            assets.extraction_ready(marker, self.cache, asset, full_hash=True)

    def test_traversal_and_special_entries_rejected(self):
        for name, kind in [("../escaped", tarfile.REGTYPE), ("/tmp/escaped", tarfile.REGTYPE),
                           (self.relative, tarfile.SYMTYPE), (self.relative, tarfile.LNKTYPE),
                           (self.relative, tarfile.CHRTYPE),
                           (f"{self.log}/{self.token}/unexpected.py", tarfile.REGTYPE)]:
            with self.subTest(name=name, kind=kind):
                archive, asset = self.archive([(name, b"danger", kind)])
                with self.assertRaises(ValueError):
                    assets.extract_archive(archive, self.cache, asset)
                self.assertFalse((self.root / "escaped").exists())
                self.assertFalse((self.cache / self.relative).exists())

    def test_existing_different_file_not_overwritten(self):
        archive, asset = self.archive([(self.relative, b"new data", tarfile.REGTYPE)])
        target = self.cache / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user existing data")
        with self.assertRaises(ValueError):
            assets.extract_archive(archive, self.cache, asset)
        self.assertEqual(target.read_bytes(), b"user existing data")
        self.assertFalse(list((self.cache / ".extracted").glob("*.json")))

    def test_symlink_destination_parent_rejected(self):
        archive, asset = self.archive([(self.relative, b"feature", tarfile.REGTYPE)])
        outside = self.root / "outside"
        outside.mkdir()
        self.cache.mkdir()
        (self.cache / self.log).symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            assets.extract_archive(archive, self.cache, asset)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
