"""Pinned official ChainFlow assets, verified downloads and resumable safe extraction.

No NAVSIM sensor data or nuPlan checkout is downloaded here.  The released VLM
cache is 1.035 TB compressed; ``plan`` reports the cost without downloading it.
Only the standard library is needed until .tar.zst extraction (zstandard).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import ssl
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "configs/navsim_chainflow_assets.json"
UPSTREAM_COMMIT = "0f3565f66546139a0183bc5807d5f7234ae2d4b0"
NAVTEST_PREFIX = "navtest_recogdrive_vlm2B_traj_cache_first_forward"
LOG_NAME = re.compile(r"^20\d\d\.\d\d\.\d\d\.[A-Za-z0-9_.-]+$")
TOKEN = re.compile(r"^[0-9a-f]{16}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another asset operation holds {path}") from None
        yield


def load_plan(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "navsim_chainflow_assets_v1" or plan.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Wrong ChainFlow asset plan schema or pinned source revision")
    seen = set()
    for asset in plan["assets"]:
        key = (asset["repo"], asset["path"])
        if key in seen or asset["role"] not in ("checkpoint", "backbone", "vlm_archive"):
            raise ValueError("Duplicate/unknown asset")
        seen.add(key)
        if (asset["repo_type"] not in ("model", "dataset")
                or not re.fullmatch(r"[0-9a-f]{40}", asset["revision"])
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
                or type(asset["size"]) is not int or asset["size"] <= 0):
            raise ValueError("Assets require pinned revision, length and SHA-256")
        relative_path(asset["path"])
    return plan


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe relative path: {name!r}")
    return path


def inside(root, relative):
    """Reject traversal and existing symlink parents outside the destination."""
    root = Path(root).resolve()
    result = root.joinpath(*relative_path(str(relative)).parts)
    if root not in result.resolve().parents:
        raise ValueError(f"Path escapes destination: {relative!r}")
    cursor = result
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Symlink in asset destination: {cursor}")
        cursor = cursor.parent
    return result


def asset_url(asset, endpoint="https://huggingface.co"):
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("https", "http") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("HF endpoint must be an HTTP(S) origin without credentials or query")
    prefix = "datasets/" if asset["repo_type"] == "dataset" else ""
    return (endpoint.rstrip("/") + "/" + prefix + asset["repo"] + "/resolve/"
            + asset["revision"] + "/" + urllib.parse.quote(asset["path"], safe="/"))


def asset_paths(root):
    root = Path(root).resolve()
    code = root / "ChainFlow-VLA"
    cache = root / "dataset/navsim/chainflow_vlm_cache"
    return {
        "checkpoint": code / "weights/stage_2/stage2_navtrain_9382.ckpt",
        "backbone": code / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors",
        "vlm_cache": cache,
        "downloads": root / "dataset/navsim/.chainflow_download",
        "manifest": cache / "assets.json",
        "release_manifest": cache / "release_manifest.json",
    }


def quarantine(path):
    path = Path(path)
    if path.exists():
        destination = path.with_name(path.name + f".bad-{time.time_ns()}")
        path.rename(destination)
        print(f"Quarantined invalid asset: {destination}", flush=True)


def _response_offset(response, offset, expected_size):
    status = getattr(response, "status", None) or response.getcode()
    if status == 200:
        effective_offset = 0  # A server ignoring Range must REPLACE, never append.
    elif status == 206:
        value = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
        if not match:
            raise ValueError(f"Missing/invalid Content-Range: {value!r}")
        start, end, total = map(int, match.groups())
        if start != offset or total != expected_size or end < start or end >= total:
            raise ValueError("Response range does not match the pinned asset/partial file")
        effective_offset = offset
    else:
        raise ValueError(f"Unexpected download HTTP status {status}")
    length = response.headers.get("Content-Length")
    if length is not None:
        length = int(length)
        if status == 200 and length != expected_size:
            raise ValueError(f"Full response has {length} bytes, expected {expected_size}")
        if status == 206 and length != end - start + 1:
            raise ValueError("Response Content-Length differs from Content-Range")
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type or "application/json" in content_type:
        raise ValueError("Server returned an HTML/JSON error document instead of a binary asset")
    return effective_offset


def download(asset, destination, *, endpoint="https://huggingface.co", insecure=False,
             attempts=12, opener=None, pause=time.sleep, timeout=60):
    """Resume only a verified HTTP range, and publish only exact-size SHA-256 files."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    ssl_context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    if opener is None:
        opener = lambda request: urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
    with lock(destination.with_name(destination.name + ".lock")):
        if destination.exists():
            if destination.stat().st_size == asset["size"] and sha256(destination) == asset["sha256"]:
                print(f"Verified existing {destination.name}", flush=True)
                return destination
            quarantine(destination)
        for attempt in range(1, attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > asset["size"]:
                quarantine(partial)
                offset = 0
            if offset == asset["size"]:
                if sha256(partial) == asset["sha256"]:
                    partial.replace(destination)
                    return destination
                quarantine(partial)
                offset = 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "ScoreFunction-ChainFlow/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"Download {destination.name}: {offset:,}/{asset['size']:,} bytes; attempt {attempt}/{attempts}", flush=True)
            try:
                request = urllib.request.Request(asset_url(asset, endpoint), headers=headers)
                with opener(request) as response:
                    actual_offset = _response_offset(response, offset, asset["size"])
                    if actual_offset != offset:
                        print("Server ignored Range; replacing the partial download from byte zero.", flush=True)
                    offset = actual_offset
                    last_report, started, start_offset = time.monotonic(), time.monotonic(), offset
                    with partial.open("ab" if offset else "wb") as stream:
                        first = True
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if first and offset == 0 and block[:512].lstrip().lower().startswith((b"<!doctype html", b"<html", b"<?xml")):
                                raise ValueError("Binary download contains an HTML/XML error page")
                            first = False
                            if offset + len(block) > asset["size"]:
                                raise ValueError("Download exceeds pinned asset size")
                            stream.write(block)
                            offset += len(block)
                            now = time.monotonic()
                            if now - last_report >= 15:
                                speed = (offset - start_offset) / max(1., now - started) / 1e6
                                print(f"  {destination.name}: {offset:,}/{asset['size']:,} ({100*offset/asset['size']:.2f}%), {speed:.2f} MB/s", flush=True)
                                last_report = now
                        stream.flush()
                if offset != asset["size"]:
                    raise IOError(f"Short transfer: {offset:,}/{asset['size']:,}; next attempt resumes")
                print(f"Checking SHA-256: {destination.name}", flush=True)
                if sha256(partial) != asset["sha256"]:
                    quarantine(partial)
                    raise ValueError("SHA-256 mismatch; retry starts with a clean partial file")
                partial.replace(destination)
                print(f"Verified {destination.name}", flush=True)
                return destination
            except (OSError, ValueError, urllib.error.URLError, socket.timeout, http.client.HTTPException) as exc:
                # Do not print response bodies or signed redirect/proxy credentials.
                detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc)
                print(f"Retry {destination.name}: {type(exc).__name__}: {detail}", flush=True)
                if attempt == attempts:
                    raise RuntimeError(f"Asset download failed after {attempts} attempts: {destination.name}") from None
                pause(min(30, attempt * 3))
    raise AssertionError("unreachable")


def cache_member_path(name):
    """Normalize only the observed upstream archive wrapper; reject arbitrary files."""
    parts = list(relative_path(name).parts)
    if parts and parts[0] == NAVTEST_PREFIX:
        parts = parts[1:]
    if not parts:
        return None
    if not LOG_NAME.fullmatch(parts[0]):
        raise ValueError(f"Unrecognized archive log folder: {name!r}")
    if len(parts) >= 2 and not TOKEN.fullmatch(parts[1]):
        raise ValueError(f"Unrecognized archive token folder: {name!r}")
    if len(parts) > 3:
        raise ValueError(f"Unexpected archive nesting: {name!r}")
    if len(parts) == 3 and parts[2] not in ("internvl_feature.gz", "trajectory_target.gz"):
        raise ValueError(f"Unexpected file in VLM archive: {name!r}")
    return PurePosixPath(*parts)


def extraction_ready(marker, root, asset, *, full_hash=False):
    marker = Path(marker)
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text())
    if data.get("archive_sha256") != asset["sha256"] or not data.get("files"):
        raise ValueError(f"Wrong extraction provenance: {marker}")
    for row in data["files"]:
        path = inside(root, row["path"])
        if not path.is_file() or path.stat().st_size != row["size"]:
            return False
        if full_hash or path.stat().st_mtime_ns != row.get("mtime_ns"):
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Extracted VLM file was changed: {path}")
    return True


def extract_archive(archive, root, asset, *, full_hash=False):
    """Stream safe members, atomically publish each file, then mark the whole shard.

    Existing different files are never overwritten. Interrupted extraction can
    restart safely; matching files are compared while reading the archive.
    """
    root, archive = Path(root).resolve(), Path(archive)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".extracted" / f"{asset['sha256']}.json"
    with lock(marker.with_suffix(".lock")):
        if extraction_ready(marker, root, asset, full_hash=full_hash):
            print(f"Extraction already verified: {archive.name}", flush=True)
            return marker
        if archive.stat().st_size != asset["size"] or sha256(archive) != asset["sha256"]:
            raise ValueError(f"Archive must pass size and SHA-256 before extraction: {archive}")
        import zstandard
        files, seen = [], set()
        print(f"Extracting {archive.name} into {root}", flush=True)
        with archive.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    relative = cache_member_path(member.name)
                    if not member.isdir() and not member.isreg():
                        raise ValueError(f"Archive links/special files are forbidden: {member.name}")
                    if relative is None:
                        if not member.isdir():
                            raise ValueError("Archive wrapper must be a directory")
                        continue
                    if member.isdir():
                        if len(relative.parts) > 2:
                            raise ValueError("Cache file path used as an archive directory")
                        continue
                    if len(relative.parts) != 3:
                        raise ValueError("Archive regular file must be log/token/feature.gz")
                    # Native cache targets are unnecessary; Score uses authoritative
                    # expert trajectories from the existing DD source manifest.
                    if relative.name == "trajectory_target.gz":
                        continue
                    if str(relative) in seen:
                        raise ValueError(f"Duplicate VLM archive member: {relative}")
                    seen.add(str(relative))
                    destination = inside(root, str(relative))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix=".extract-", dir=destination.parent)
                    digest, count = hashlib.sha256(), 0
                    try:
                        stream = tar.extractfile(member)
                        if stream is None:
                            raise ValueError(f"Cannot read archive member: {member.name}")
                        with os.fdopen(fd, "wb") as output, stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                output.write(block)
                                digest.update(block)
                                count += len(block)
                        if count != member.size:
                            raise ValueError(f"Truncated archive member: {member.name}")
                        digest = digest.hexdigest()
                        if destination.exists():
                            if destination.stat().st_size != count or sha256(destination) != digest:
                                raise ValueError(f"Existing VLM cache differs; left untouched: {destination}")
                        else:
                            os.replace(temporary, destination)
                        files.append({"path": str(relative), "size": count, "sha256": digest,
                                      "mtime_ns": destination.stat().st_mtime_ns})
                        if len(files) % 100 == 0:
                            print(f"  {archive.name}: {len(files):,} VLM samples extracted", flush=True)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
        if not files:
            raise ValueError(f"No VLM features found in archive: {archive}")
        atomic_json(marker, {"schema": "chainflow_vlm_extracted_v1", "archive_sha256": asset["sha256"],
                             "archive_size": asset["size"], "files": files})
        print(f"Ready {archive.name}: {len(files):,} VLM samples", flush=True)
        return marker


def selected_assets(plan, only="all", scope="all"):
    selected = [asset for asset in plan["assets"] if only == "all"
            or only == "weights" and asset["role"] in ("checkpoint", "backbone")
            or only == "navtest" and asset["role"] == "vlm_archive" and asset["path"].startswith("navtest/")
            or only == "trainval" and asset["role"] == "vlm_archive" and asset["path"].startswith("trainval/")]
    return [a for a in selected if scope == "all" or a["role"] != "vlm_archive"
            or a["path"].startswith("navtest/")]


def publish_release_manifest(path, plan):
    """Stable conditioner identity, including ALL assets regardless of download scope.

    Do not add status, local paths, timestamps, extraction state or endpoint here.
    The separate assets.json inventory records mutable completion information.
    """
    release = {"schema": "chainflow_release_manifest_v1",
               "upstream_repository": plan["upstream_repository"],
               "upstream_commit": plan["upstream_commit"],
               "training_split": plan["training_split"],
               "assets": sorted(plan["assets"], key=lambda a: (a["repo"], a["path"]))}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != release:
            raise ValueError(f"Existing release provenance differs; left untouched: {path}")
    else:
        atomic_json(path, release)
    return path


def fetch(plan, root, *, only="all", scope="native", endpoint="https://huggingface.co", insecure=False,
          attempts=12, keep_archives=False, full_hash=False):
    paths = asset_paths(root)
    paths["vlm_cache"].mkdir(parents=True, exist_ok=True)
    with lock(paths["vlm_cache"] / ".assets.lock"):
        publish_release_manifest(paths["release_manifest"], plan)
        for asset in selected_assets(plan, only, scope):
            if asset["role"] == "vlm_archive":
                marker = paths["vlm_cache"] / ".extracted" / f"{asset['sha256']}.json"
                if extraction_ready(marker, paths["vlm_cache"], asset, full_hash=full_hash):
                    print(f"REUSED extracted {asset['path']}", flush=True)
                    continue
                destination = inside(paths["downloads"], asset["path"])
            else:
                destination = paths[asset["role"]]
            download(asset, destination, endpoint=endpoint, insecure=insecure, attempts=attempts)
            if asset["role"] == "vlm_archive":
                extract_archive(destination, paths["vlm_cache"], asset, full_hash=full_hash)
                if not keep_archives:
                    destination.unlink()
                    print(f"Removed verified extracted archive: {destination.name}", flush=True)
        inventory = []
        for asset in plan["assets"]:
            row = dict(asset)
            if asset["role"] == "vlm_archive":
                marker = paths["vlm_cache"] / ".extracted" / f"{asset['sha256']}.json"
                row.update(extraction_marker=str(marker), ready=extraction_ready(marker, paths["vlm_cache"], asset))
            else:
                path = paths[asset["role"]]
                row.update(local_path=str(path), ready=path.is_file() and path.stat().st_size == asset["size"]
                           and sha256(path) == asset["sha256"])
            inventory.append(row)
        atomic_json(paths["manifest"], {"schema": "chainflow_local_assets_v1", "upstream_commit": UPSTREAM_COMMIT,
                    "training_split": plan["training_split"], "complete": all(a["ready"] for a in inventory),
                    "native_complete": all(a["ready"] for a in selected_assets({"assets": inventory}, scope="native")),
                    "release_manifest": str(paths["release_manifest"]),
                    "checkpoint": str(paths["checkpoint"]), "backbone": str(paths["backbone"]),
                    "vlm_cache": str(paths["vlm_cache"]), "assets": inventory})
        print(f"Asset inventory: {paths['manifest']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "fetch"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--only", choices=("all", "weights", "navtest", "trainval"), default="all")
    parser.add_argument("--scope", choices=("native", "all"), default="native",
                        help="native: checkpoint, DINO and navtest VLM (~109 GB); all: also trainval VLM (~1.035 TB)")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for this download process only")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--verify-extracted", action="store_true", help="Rehash every extracted feature instead of using recorded size/mtime")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    if args.only == "trainval" and args.scope != "all":
        parser.error("--only trainval requires --scope all")
    plan = load_plan(args.manifest)
    selected = selected_assets(plan, args.only, args.scope)
    total = sum(a["size"] for a in selected)
    print(f"Selected {len(selected)} official assets: {total:,} bytes ({total / 1e12:.3f} TB compressed).", flush=True)
    print("Existing NAVSIM data/maps are reused. VLM extraction additionally needs its expanded feature storage.", flush=True)
    if args.action == "plan":
        print(json.dumps({"paths": {k: str(v) for k, v in asset_paths(args.root).items()}, "assets": selected}, indent=2))
        return
    fetch(plan, args.root, only=args.only, scope=args.scope, endpoint=args.endpoint, insecure=args.insecure,
          attempts=args.attempts, keep_archives=args.keep_archives, full_hash=args.verify_extracted)


if __name__ == "__main__":
    main()
