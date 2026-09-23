"""Download the public OpenPack 1.0.0 archives with publisher checksums."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import re
import urllib.request

METADATA_URL = "https://zenodo.org/api/records/8145223"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"
logger = logging.getLogger(__name__)


def download_archive(entry: dict, output: Path) -> dict:
    """Verify existing archives or atomically finish a checked .part download."""
    name = entry["key"]
    if not re.fullmatch(r"U0[12]\d{2}\.zip", name):
        raise ValueError(f"Unexpected subject archive: {name}")
    algorithm, expected = entry["checksum"].split(":", 1)
    if algorithm != "md5" or not re.fullmatch(r"[0-9a-f]{32}", expected):
        raise ValueError(f"Unsupported publisher checksum: {name}")
    destination = output / name
    partial = destination.with_suffix(".zip.part")
    if not destination.exists():
        logger.info("OpenPack downloading archive=%s bytes=%d", name, entry["size"])
        url = f"https://zenodo.org/api/records/8145223/files/{name}/content"
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    candidate = destination if destination.exists() else partial
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with candidate.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    if candidate.stat().st_size != entry["size"] or md5.hexdigest() != expected:
        raise ValueError(f"OpenPack archive size/checksum mismatch: {candidate}")
    if candidate == partial:
        partial.rename(destination)
    logger.info("OpenPack verified archive=%s", name)
    return {"name": name, "bytes": entry["size"], "md5": expected, "sha256": sha256.hexdigest()}


def main() -> None:
    """Acquire all 21 source IDs with explicit provider and integrity provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Previously retained Zenodo record 8145223 JSON")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between one and four")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.metadata:
        raw = args.metadata.read_bytes()
    else:
        with urllib.request.urlopen(METADATA_URL, timeout=30) as response:
            raw = response.read()
    metadata = json.loads(raw)
    if int(metadata["id"]) != 8145223:
        raise ValueError("Expected the OpenPack 1.0.0 Zenodo record")
    entries = sorted(
        (e for e in metadata["files"] if re.fullmatch(r"U0[12]\d{2}\.zip", e["key"])), key=lambda e: e["key"]
    )
    expected = {f"U01{i:02d}.zip" for i in range(1, 12)} | {f"U02{i:02d}.zip" for i in range(1, 11)}
    if {e["key"] for e in entries} != expected or len(entries) != len(expected):
        raise ValueError("Publisher metadata does not contain the complete expected subject inventory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "zenodo_8145223.json").write_bytes(raw)
    (args.output / "LICENSE_SOURCE.txt").write_text(
        "OpenPack Dataset 1.0.0 — Naoya Yoshimura, Jaime Morales, Takuya Maekawa and Takahiro Hara\n"
        "Creative Commons Attribution Non Commercial Share Alike 4.0 International (CC-BY-NC-SA-4.0)\n"
        f"License terms: {LICENSE_URL}\nSource release: https://doi.org/10.5281/zenodo.8145223\n"
        "Dataset paper: https://doi.org/10.1109/PerCom59722.2024.10494448\n"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Bound pending requests as well as running requests; stop at a failed batch.
        verified = []
        for start in range(0, len(entries), args.workers):
            verified.extend(pool.map(lambda e: download_archive(e, args.output), entries[start : start + args.workers]))
    manifest = {
        "release": "1.0.0",
        "metadata_url": METADATA_URL,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC-BY-NC-SA-4.0",
        "license_url": LICENSE_URL,
        "archives": verified,
    }
    temp = args.output / "source_manifest.json.part"
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output / "source_manifest.json")


if __name__ == "__main__":
    main()
