#!/usr/bin/env python3
"""Download the official Flow Planner weights and matching configuration.

The official DiffusionAD/Flow-Planner README links ttwhy/flow-planner on
Hugging Face. Both files are pinned to that repository's immutable revision
and to bytes independently downloaded and checked during integration.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import ssl
import tempfile
import urllib.parse
import urllib.request


SOURCE_REPOSITORY = "https://github.com/DiffusionAD/Flow-Planner"
MODEL_REPOSITORY = "https://huggingface.co/ttwhy/flow-planner"
REVISION = "b627934a8db3021892afd13f5278ba0149532efc"
ASSETS = {
    "model.pth": {
        "size_bytes": 57244066,
        "sha256": "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e",
    },
    "model_config.yaml": {
        "size_bytes": 7036,
        "sha256": "2914c4f174e7fba8baf9ee7d42300c640a285f9d6fa13d36b78f0a3cbb9a03fe",
    },
}


def verified(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == asset["sha256"]


def download_asset(output_dir, filename, context, timeout=120):
    asset = ASSETS[filename]
    output = Path(output_dir) / filename
    url = f"{MODEL_REPOSITORY}/resolve/{REVISION}/{filename}"
    if output.exists() or output.is_symlink():
        if not verified(output, asset):
            raise ValueError(f"Existing file differs from the official Flow Planner asset; preserved: {output}")
        print(f"Already verified: {output}", flush=True)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    print(f"Downloading official {filename} ({asset['size_bytes']:,} bytes)", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Flow-Planner-checkpoint-download/1.0"})
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise ValueError("Download redirected to a non-HTTPS endpoint")
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type or (filename.endswith(".pth") and "text/" in content_type):
                raise ValueError(f"{filename}: download returned a webpage/text response")
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=filename + ".", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if total == 0:
                        beginning = block[:1024].lstrip().lower()
                        if beginning.startswith((b"<!doctype html", b"<html")):
                            raise ValueError(f"{filename}: download returned an HTML page")
                    total += len(block)
                    if total > asset["size_bytes"]:
                        raise ValueError(f"{filename}: download is larger than the pinned asset")
                    stream.write(block)

        if not verified(temporary, asset):
            raise ValueError(f"{filename}: downloaded size/SHA256 differs from the official pinned asset")
        # Publish without replacing any destination created by another process.
        os.link(temporary, output)
        metadata = {
            "filename": filename,
            "source_repository": SOURCE_REPOSITORY,
            "model_repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "download_url": url,
            **asset,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        output.with_suffix(output.suffix + ".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Downloaded and verified: {output}\nSHA256 {asset['sha256']}", flush=True)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(output_dir, ca_bundle=None, timeout=120):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    context = ssl.create_default_context()
    if ca_bundle is not None:
        # Add an explicitly supplied corporate CA without disabling verification.
        context.load_verify_locations(cafile=str(Path(ca_bundle).expanduser().resolve()))
    return [download_asset(output_dir, name, context, timeout) for name in ASSETS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/home/data/wyk/Flow-Planner/checkpoints")
    parser.add_argument("--ca-bundle", help="Additional trusted CA certificates in PEM format")
    parser.add_argument("--timeout", type=float, default=120, help="Network socket timeout in seconds")
    args = parser.parse_args()
    try:
        download(args.output_dir, args.ca_bundle, args.timeout)
    except Exception as error:
        raise SystemExit(
            f"Flow Planner download failed: {error}\n"
            "If your company uses a private TLS certificate, pass its PEM CA bundle with --ca-bundle. "
            "If the network blocks Hugging Face, download the same pinned files on an accessible machine "
            "and copy them to --output-dir; rerunning verifies their hashes without downloading again."
        ) from error


if __name__ == "__main__":
    main()
