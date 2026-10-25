#!/usr/bin/env python3
"""Fetch and verify the public ForSim copy of PLUTO's original IL checkpoint.

The original PLUTO README's OneDrive link returned HTTP 404 on 2026-10-25.
ForSim's public README links this exact file as its pretrained PLUTO checkpoint:
https://github.com/CurryChen77/ForSim#data-and-ckpt
This is a secondary copy, not a newly trained or RL-fine-tuned model. The hash
pins the bytes inspected during integration; the unavailable OneDrive original
could not be independently compared byte-for-byte.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1rI_we4zkk3Jk6wr7IzF-nMBnRMJsuOGT&export=download&confirm=t"
)
CHECKPOINT_SIZE = 51431136
CHECKPOINT_SHA256 = "ce60d7c854d10b310c4e1705099d799a6ea008baa4288e668626f4a2f22943e1"


def verified(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return Path(path).stat().st_size == CHECKPOINT_SIZE and digest.hexdigest() == CHECKPOINT_SHA256


def download(output, url=CHECKPOINT_URL):
    output = Path(output).expanduser().resolve()
    if output.exists():
        if not verified(output):
            raise ValueError(f"Existing file differs from the verified PLUTO checkpoint; preserved: {output}")
        print(f"Checkpoint already verified: {output}", flush=True)
        return output
    if not url.startswith("https://"):
        raise ValueError("Checkpoint URL must use HTTPS")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PLUTO-checkpoint-download/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/" in content_type or "html" in content_type:
                raise ValueError("Download returned a webpage, not checkpoint bytes")
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > CHECKPOINT_SIZE:
                        raise ValueError("Download is larger than the verified checkpoint")
                    stream.write(block)
        if not verified(temporary):
            raise ValueError("Downloaded size/SHA256 differs from the verified checkpoint")
        # A concurrent download must not overwrite a newly created destination.
        os.link(temporary, output)
        metadata = {
            "checkpoint": "pluto_1M_aux_cil.ckpt",
            "source_repository": "https://github.com/CurryChen77/ForSim",
            "source_folder": "https://drive.google.com/drive/folders/18bXqs8nT1gr3-xnTUlxGN2jxnQV5BqFQ",
            "download_url": url,
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": CHECKPOINT_SIZE,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": "Public secondary copy; original OneDrive link was unavailable for byte comparison.",
        }
        output.with_suffix(output.suffix + ".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Downloaded and verified: {output}\nSHA256 {CHECKPOINT_SHA256}", flush=True)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/home/data/wyk/pluto/checkpoints/pluto_1M_aux_cil.ckpt")
    parser.add_argument("--url", default=CHECKPOINT_URL,
                        help="Alternate HTTPS endpoint for the exact same hash-pinned checkpoint")
    args = parser.parse_args()
    try:
        download(args.output, args.url)
    except Exception as error:
        raise SystemExit(
            f"PLUTO download failed: {error}\n"
            "The verified endpoint is public, but this machine's network may block it. "
            "Download the same file on an accessible machine and copy it to --output; "
            "rerunning this command verifies its SHA256 without downloading again."
        ) from error


if __name__ == "__main__":
    main()
