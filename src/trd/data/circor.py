"""Reproducible CirCor preparation with linked identities and original intervals."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import wave
import zipfile

import numpy as np

from trd.data.temporal import Interval, validate_intervals

logger = logging.getLogger("PMT")
SOURCE_URL = "https://physionet.org/files/circor-heart-sound/1.0.3/"
ARCHIVE_URL = "https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/"
CLASS_NAMES = ["S1", "systole", "S2", "diastole"]


def file_sha256(path: Path) -> str:
    """Hash a source/derived file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_circor_archive(archive: Path, destination: Path) -> Path:
    """Extract an official archive into a new directory and return its release root."""
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {item.filename}")
        destination.mkdir(parents=True)
        handle.extractall(destination)
    roots = list(destination.glob("*/training_data.csv"))
    if (destination / "training_data.csv").is_file():
        roots.append(destination / "training_data.csv")
    if len(roots) != 1:
        raise ValueError(f"Expected one CirCor release root in {destination}")
    return roots[0].parent


def identity_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    """Union every Additional ID link; reject duplicate or unresolved public IDs."""
    ids = [str(row["Patient ID"]).strip() for row in rows]
    if len(set(ids)) != len(ids) or not ids or any(not re.fullmatch(r"\d+", v) for v in ids):
        raise ValueError("Patient IDs must be unique nonempty numeric identifiers")
    parent = {key: key for key in ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row, patient in zip(rows, ids):
        additional = str(row.get("Additional ID", "")).strip()
        if additional in {"", "nan", "NaN", "NA"}:
            continue
        if additional not in parent:
            raise ValueError(f"Additional ID {additional!r} for {patient} is absent from public metadata")
        left, right = find(patient), find(additional)
        parent[max(left, right)] = min(left, right)
    return {key: find(key) for key in sorted(ids)}


def partition_groups(groups: dict[str, str], *, seed: int) -> dict[str, str]:
    """Make a fixed 70/15/15 linked-group split independent of encoder seeds."""
    unique = sorted(set(groups.values()))
    if len(unique) < 7 or seed < 0:
        raise ValueError("Preparation needs at least seven independent groups and a nonnegative data seed")
    random.Random(seed).shuffle(unique)
    train_end = math.floor(0.70 * len(unique))
    val_end = train_end + math.floor(0.15 * len(unique))
    split = {key: ("train" if i < train_end else "val" if i < val_end else "test") for i, key in enumerate(unique)}
    return {patient: split[group] for patient, group in groups.items()}


@dataclass(frozen=True)
class CirCorAnnotations:
    """Audited targets plus unchanged source rows for a single recording."""

    intervals: list[Interval]
    events: list[Interval]
    source_rows: list[tuple[float, float, int]]
    audit: dict[str, int | float | bool]


def read_circor_annotations(path: Path, *, duration: float, rate: int = 4000) -> CirCorAnnotations:
    """Retain source rows and apply the explicit CirCor annotation-mask protocol.

    Ignore zero-duration rows, clip at most one sample of endpoint rounding,
    mask contradictory overlap, and quarantine labels when timestamps restart.
    Event references keep original onsets rather than new mask boundaries.
    """
    intervals: list[Interval] = []
    source_rows = []
    audit = {
        "zero_duration_rows": 0,
        "clipped_endpoints": 0,
        "overlap_rows": 0,
        "nonmonotonic_rows": 0,
        "conflict_seconds": 0.0,
        "quarantined": False,
        "zero_duration_only": False,
    }
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 3:
                raise ValueError(f"Malformed annotation row {path}:{line}")
            start, end = float(row[0]), float(row[1])
            state = int(row[2])
            source_rows.append((start, end, state))
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < start
                or end > duration + 1 / rate
            ):
                raise ValueError(f"Invalid CirCor annotation bounds in {path}:{line}: {row}")
            if start == end:
                audit["zero_duration_rows"] += 1
                continue  # Includes the published [0,0,28] row: it has no temporal support.
            if state not in range(5):
                raise ValueError(f"Invalid CirCor state {state} in {path}:{line}")
            if start >= duration:
                raise ValueError(f"CirCor annotation starts outside the recording: {path}:{line}")
            if intervals:
                audit["overlap_rows"] += start < intervals[-1].end - 1e-6
                audit["nonmonotonic_rows"] += start < intervals[-1].start
            if end > duration:
                audit["clipped_endpoints"] += 1
                end = duration
            intervals.append(Interval(start, end, state - 1))
    if not intervals:
        if not source_rows:
            raise ValueError(f"Empty annotation file: {path}")
        audit["zero_duration_only"] = True
        return CirCorAnnotations([Interval(0, duration, -1)], [], source_rows, audit)
    if audit["nonmonotonic_rows"]:
        audit["quarantined"] = True
        return CirCorAnnotations([Interval(0, duration, -1)], [], source_rows, audit)

    changes: dict[float, list[tuple[int, int]]] = {0.0: [], duration: []}
    for interval in intervals:
        changes.setdefault(interval.start, []).append((interval.label, 1))
        changes.setdefault(interval.end, []).append((interval.label, -1))
    boundaries = sorted(changes)
    active: Counter[int] = Counter()
    targets: list[Interval] = []
    for left, right in zip(boundaries, boundaries[1:]):
        for label, delta in changes[left]:
            active[label] += delta
        labels = [label for label, count in active.items() if count > 0]
        label = labels[0] if len(labels) == 1 else -1
        if len(labels) > 1:
            audit["conflict_seconds"] += right - left
        if targets and targets[-1].label == label:
            targets[-1] = Interval(targets[-1].start, right, label)
        else:
            targets.append(Interval(left, right, label))
    validate_intervals(targets, duration=duration, num_classes=4)

    events = []
    for label in range(4):
        merged: list[Interval] = []
        for interval in (a for a in intervals if a.label == label):
            if merged and interval.start <= merged[-1].end:
                merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end), label)
            else:
                merged.append(interval)
        events.extend(merged)
    return CirCorAnnotations(targets, sorted(events, key=lambda a: a.start), source_rows, audit)


def _verify_release(source: Path) -> dict[str, str]:
    checksum_file = source / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in checksum_file.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Malformed release checksum line: {line}")
        if name in expected:
            raise ValueError(f"Duplicate checksum entry: {name}")
        expected[name] = checksum
    if not expected:
        raise ValueError("Empty source checksum manifest")
    for i, (name, checksum) in enumerate(expected.items()):
        path = source / name
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"Missing or corrupt release file: {path}")
        if (i + 1) % 2000 == 0:
            logger.info("CirCor source checksum verification: %d/%d", i + 1, len(expected))
    return expected


def prepare_circor(source: Path, output: Path, *, data_seed: int = 20260912) -> dict:
    """Verify a complete release and atomically publish normalized recordings/manifest.

    Derived waveforms are float32 [samples,1], normalized from each complete
    recording without labels. The retained original archive remains authoritative.
    A failed preparation leaves only the explicit .preparing directory.
    """
    source, output = Path(source), Path(output)
    staging = output.with_name(output.name + ".preparing")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to replace prepared data or previous staging directory: {output}")
    expected = _verify_release(source)
    for name in ("training_data.csv", "RECORDS", "LICENSE.txt"):
        if name not in expected:
            raise ValueError(f"Release checksums do not cover required metadata: {name}")
    with (source / "training_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = identity_groups(rows)
    splits = partition_groups(groups, seed=data_seed)
    records = [line.strip() for line in (source / "RECORDS").read_text().splitlines() if line.strip()]
    if len(set(records)) != len(records):
        raise ValueError("Duplicate RECORDS entry")
    observed_wav = {str(p.relative_to(source).with_suffix("")) for p in (source / "training_data").glob("*.wav")}
    if set(records) != observed_wav:
        raise ValueError("RECORDS and waveform inventory differ")
    (staging / "waveforms").mkdir(parents=True)
    prepared = []
    for i, relative in enumerate(records):
        if not re.fullmatch(r"training_data/\d+_[A-Za-z0-9_]+", relative):
            raise ValueError(f"Unexpected recording name: {relative}")
        stem = Path(relative).name
        patient = stem.split("_", 1)[0]
        if patient not in groups:
            raise ValueError(f"Recording patient absent from metadata: {stem}")
        for extension in (".wav", ".hea", ".tsv"):
            if relative + extension not in expected:
                raise ValueError(f"Recording lacks verified source component: {relative}{extension}")
        wav_path = source / (relative + ".wav")
        with wave.open(str(wav_path), "rb") as handle:
            rate, count = handle.getframerate(), handle.getnframes()
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or rate != 4000 or count < 1:
                raise ValueError(f"Expected mono 4 kHz PCM16 recording: {stem}")
            waveform = np.frombuffer(handle.readframes(count), dtype="<i2").astype(np.float32)
        header = (source / (relative + ".hea")).read_text().splitlines()[0].split()
        if len(header) < 4 or (int(header[1]), float(header[2]), int(header[3])) != (1, rate, count):
            raise ValueError(f"WFDB/WAV dimensions disagree: {stem}")
        if waveform.size != count:
            raise ValueError(f"Truncated waveform: {stem}")
        mean, std = float(waveform.mean(dtype=np.float64)), float(waveform.std(dtype=np.float64))
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Constant or nonfinite waveform: {stem}")
        waveform = ((waveform - mean) / std).astype(np.float32)
        annotation = read_circor_annotations(source / (relative + ".tsv"), duration=count / rate, rate=rate)
        annotations = annotation.intervals
        target = Path("waveforms") / (stem + ".npy")
        np.save(staging / target, waveform[:, None], allow_pickle=False)
        durations = [sum(a.end - a.start for a in annotations if a.label == c) for c in range(4)]
        unusual = sum(
            a.label >= 0 and b.label >= 0 and a.label != b.label and b.label != (a.label + 1) % 4
            for a, b in zip(annotations, annotations[1:])
        )
        prepared.append(
            {
                "record_id": stem,
                "group_id": groups[patient],
                "patient_id": patient,
                "split": splits[patient],
                "path": str(target),
                "num_samples": count,
                "sha256": file_sha256(staging / target),
                "normalization_mean_pcm": mean,
                "normalization_std_pcm": std,
                "annotations": [[a.start, a.end, a.label] for a in annotations],
                "event_annotations": [[a.start, a.end, a.label] for a in annotation.events],
                "source_annotations": annotation.source_rows,
                "annotation_audit": annotation.audit,
                "labeled_seconds": sum(durations),
                "class_seconds": durations,
                "unusual_transitions": unusual,
            }
        )
        if (i + 1) % 250 == 0:
            logger.info("CirCor prepared recordings: %d/%d", i + 1, len(records))
    if {r["patient_id"] for r in prepared} != set(groups):
        raise ValueError("Patient metadata and recording inventory differ")
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in prepared if r["split"] == split]
        summary[split] = {
            "groups": len({r["group_id"] for r in subset}),
            "records": len(subset),
            "signal_seconds": sum(r["num_samples"] / 4000 for r in subset),
            "labeled_seconds": sum(r["labeled_seconds"] for r in subset),
            "records_without_labels": sum(r["labeled_seconds"] == 0 for r in subset),
            "quarantined_annotation_records": sum(r["annotation_audit"]["quarantined"] for r in subset),
            "conflicting_annotation_seconds": sum(r["annotation_audit"]["conflict_seconds"] for r in subset),
            "class_seconds": [sum(r["class_seconds"][c] for r in subset) for c in range(4)],
        }
    manifest = {
        "format_version": 1,
        "dataset": "CIRCOR",
        "source_version": "1.0.3",
        "source_url": SOURCE_URL,
        "protocol": "circor_linked_identity_70_15_15_v1",
        "data_seed": data_seed,
        "sampling_rate_hz": 4000,
        "channel_names": ["PCG"],
        "class_names": CLASS_NAMES,
        "normalization": "record_demean_std_v1",
        "interval_convention": "start_inclusive_end_exclusive_seconds",
        "annotation_policy": "circor_annotation_masks_v1",
        "source_checksums_manifest_sha256": file_sha256(source / "SHA256SUMS.txt"),
        "source_files": expected,
        "identity_groups": groups,
        "summary": summary,
        "records": prepared,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    staging.rename(output)
    logger.info("CirCor preparation complete: %s", summary)
    return manifest
