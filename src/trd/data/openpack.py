"""Prepare verified OpenPack wrist recordings with linked people and native clocks."""

from __future__ import annotations

import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import io
import json
import logging
from pathlib import Path
import random
import re
import zipfile

import numpy as np
from scipy.signal import resample_poly

from trd.data.temporal import Interval, validate_intervals

logger = logging.getLogger("PMT")
SOURCE_REVISION = "0485a97c508c74018320dfc3eb90a1b6bc2d7d54"
SOURCE_USERS = tuple([f"U01{i:02d}" for i in range(1, 12)] + [f"U02{i:02d}" for i in range(1, 11)])
PERSON_ALIASES = {"U0202": "U0105", "U0203": "U0108", "U0204": "U0110", "U0205": "U0107", "U0210": "U0103"}
U0108_S0400_IMU_SHA256 = "120914f4fc56f2b64be0d6619be9935d2d165f903696e3c116b20f017db92814"
CLASS_NAMES = [
    "Picking",
    "Relocate Item Label",
    "Assemble Box",
    "Insert Items",
    "Close Box",
    "Attach Box Label",
    "Scan Label",
    "Attach Shipping Label",
    "Put on Back Table",
    "Fill out Order",
]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file in bounded chunks without loading an archive into RAM."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def operation_index(value: str | int) -> int:
    """Map release operation IDs to ten classes; Null is never Picking."""
    value = int(value)
    if value in {8100, 8200}:
        return -1
    if value not in range(100, 1001, 100):
        raise ValueError(f"Unknown OpenPack operation ID: {value}")
    return value // 100 - 1


def person_splits(users: list[str], *, seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Split linked people approximately 60/20/20, keeping exposed U0209 in train."""
    if seed < 0 or not users or set(users) - set(SOURCE_USERS) or len(set(users)) != len(users):
        raise ValueError("Invalid OpenPack users or split seed")
    groups = {user: PERSON_ALIASES.get(user, user) for user in sorted(users)}
    people = sorted(set(groups.values()))
    if len(people) < 7:
        raise ValueError("OpenPack research split needs at least seven independent people")
    exposed = ["U0209"] if "U0209" in people else []
    pool = [p for p in people if p not in exposed]
    random.Random(seed).shuffle(pool)
    ordered = exposed + pool
    held_out = max(1, round(len(people) * 0.2))
    training_end, validation_end = len(people) - 2 * held_out, len(people) - held_out
    split = {p: "train" if i < training_end else "val" if i < validation_end else "test" for i, p in enumerate(ordered)}
    return groups, {user: split[person] for user, person in groups.items()}


def resample_acceleration(
    timestamps: np.ndarray, acceleration: np.ndarray, *, rate: int = 30
) -> tuple[np.ndarray, dict]:
    """Regularize small timestamp jitter, anti-alias to 30 Hz, then clip/scale G.

    Large gaps fail rather than creating motion across unobserved time. Native
    clocks are retained separately. Output sample zero is the first native time.
    """
    timestamps = np.asarray(timestamps)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or timestamps.dtype.kind not in "iu"
        or len(timestamps) < 3
        or acceleration.shape != (len(timestamps), 3)
        or not np.isfinite(acceleration).all()
    ):
        raise ValueError("Expected finite acceleration [samples,3] and integer millisecond timestamps")
    delta = np.diff(timestamps)
    if np.any(delta <= 0):
        raise ValueError("Duplicate or nonmonotone OpenPack timestamps")
    period = int(np.median(delta))
    if not 10 <= period <= 100 or delta.max() > 1.5 * period or delta.min() < 0.5 * period:
        raise ValueError("Unsupported OpenPack cadence or a signal gap; do not interpolate missing motion")
    if rate != 30:
        raise ValueError("OpenPack prepared_v1 fixes the shared signal grid at 30 Hz")
    relative = timestamps - timestamps[0]
    regular_times = np.arange(int(relative[-1] // period) + 1, dtype=np.int64) * period
    regular = np.stack([np.interp(regular_times, relative, acceleration[:, c]) for c in range(3)], axis=1)
    ratio = Fraction(rate * period, 1000)
    values = resample_poly(regular, ratio.numerator, ratio.denominator, axis=0, padtype="line")
    count = int(regular_times[-1] * rate // 1000) + 1
    values = values[:count]
    audit = {
        "native_samples": len(timestamps),
        "native_period_ms": period,
        "timestamp_delta_counts": {str(int(d)): int(n) for d, n in zip(*np.unique(delta, return_counts=True))},
        "irregular_steps": int(np.count_nonzero(delta != period)),
        "clipped_values": int(np.count_nonzero(np.abs(values) > 3)),
        "polyphase_up": ratio.numerator,
        "polyphase_down": ratio.denominator,
    }
    return ((np.clip(values, -3, 3) + 3) / 6).astype(np.float32), audit


def _csv_rows(data: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("Empty OpenPack CSV")
    return rows


def _annotation_rows(data: bytes, *, user: str, session: str, one_hz: bool) -> list[list]:
    result = []
    for row in _csv_rows(data):
        if row["user"] != user or row["session"] != session:
            raise ValueError("Annotation identity does not match its recording")
        label = operation_index(row["id"])
        if one_hz:
            time = int(row["unixtime"])
            if time % 1000 or (result and time - result[-1][0] != 1000):
                raise ValueError("OpenPack 1 Hz labels must have unique contiguous wall-clock seconds")
            result.append([time, label])
        else:
            start, end = (datetime.fromisoformat(row[k]) for k in ("start", "end"))
            if start.tzinfo is None or end.tzinfo is None or end <= start:
                raise ValueError("Operation intervals require ordered timezone-aware timestamps")
            a, b = round(start.timestamp() * 1000), round(end.timestamp() * 1000)
            if result and a < result[-1][1]:
                raise ValueError("Overlapping or unordered OpenPack source intervals")
            result.append([a, b, label, row["box"]])
    return result


def _prepare_record(
    archive: zipfile.ZipFile, paths: dict[str, str], *, user: str, session: str, group: str, split: str, staging: Path
) -> dict:
    raw = {kind: archive.read(path) for kind, path in paths.items()}
    rows = _csv_rows(raw["imu"])
    source_count, source_origin = len(rows), int(rows[0]["unixtime"])
    correction = user == "U0101" and session == "S0500"
    removed, reason, gap_steps = 0, None, []
    if correction:
        rows = rows[1:]  # Pinned toolkit's atr01 timestamp-alignment correction.
        removed, reason = 1, "pinned_toolkit_U0101_S0500_first_row"
    if user == "U0108" and session == "S0400":
        if hashlib.sha256(raw["imu"]).hexdigest() != U0108_S0400_IMU_SHA256:
            raise ValueError("U0108-S0400 prefix correction requires the audited exact source CSV")
        # Timestamp-only audit: two initial packet-loss gaps, then a regular suffix.
        # Keep the original CSV and full annotation tracks; never fill missing motion.
        rows = rows[245:]
        removed, reason = 245, "pmt_pinned_U0108_S0400_initial_gaps_v1"
        gap_steps = [{"after_native_row": 243, "delta_ms": 1260}, {"after_native_row": 244, "delta_ms": 5880}]
    timestamps = np.array([int(row["unixtime"]) for row in rows], dtype=np.int64)
    acceleration = np.array([[float(row[c]) for c in ("acc_x", "acc_y", "acc_z")] for row in rows])
    signal, audit = resample_acceleration(timestamps, acceleration)
    audit["source_first_row_removed"] = correction
    audit["source_native_samples"] = source_count
    audit["source_prefix_correction"] = (
        {
            "reason": reason,
            "removed_samples": removed,
            "source_origin_unix_ms": source_origin,
            "retained_origin_unix_ms": int(timestamps[0]),
            "removed_elapsed_ms": int(timestamps[0]) - source_origin,
            "gap_steps": gap_steps,
        }
        if removed
        else None
    )
    labels = _annotation_rows(raw["labels"], user=user, session=session, one_hz=True)
    intervals = _annotation_rows(raw["intervals"], user=user, session=session, one_hz=False)
    origin, duration = int(timestamps[0]), len(signal) / 30
    targets = []
    for time, label in labels:
        a, b = max(0.0, (time - origin) / 1000), min(duration, (time + 1000 - origin) / 1000)
        if b > a:
            targets.append(Interval(a, b, label))
    validate_intervals(targets, duration=duration, num_classes=10)
    record_id = f"{user}-{session}"
    relative = Path("waveforms") / f"{record_id}.npy"
    np.save(staging / relative, signal, allow_pickle=False)
    # Keep native clocks, rather than treating the prepared 30 Hz row count as source timing.
    time_path = Path("timestamps") / f"{record_id}.npy"
    np.save(staging / time_path, timestamps, allow_pickle=False)
    last_per_second = np.r_[timestamps[1:] // 1000 != timestamps[:-1] // 1000, True]
    class_seconds = [sum(a.end - a.start for a in targets if a.label == c) for c in range(10)]
    return {
        "record_id": record_id,
        "group_id": group,
        "user_id": user,
        "session_id": session,
        "split": split,
        "path": str(relative),
        "num_samples": len(signal),
        "sha256": file_digest(staging / relative),
        "origin_unix_ms": origin,
        "native_timestamps_path": str(time_path),
        "native_timestamps_sha256": file_digest(staging / time_path),
        "native_last_timestamps_ms": timestamps[last_per_second].tolist(),
        "annotations": [[a.start, a.end, a.label] for a in targets],
        "operation_labels_1hz": labels,
        "source_operation_intervals": intervals,
        "labeled_seconds": sum(class_seconds),
        "class_seconds": class_seconds,
        "source_files": {paths[k]: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        "signal_audit": audit,
    }


def prepare_openpack(source: Path, output: Path, *, data_seed: int = 20260913) -> dict:
    """Verify all downloaded subject archives and atomically publish prepared_v1."""
    source, output = Path(source), Path(output)
    staging = output.with_name(output.name + ".preparing")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to replace prepared data or staging: {output}")
    provenance = json.loads((source / "source_manifest.json").read_text())
    entries = provenance["archives"]
    if (
        provenance["release"] != "1.0.0"
        or len(entries) != len(SOURCE_USERS)
        or {e["name"] for e in entries} != {f"{u}.zip" for u in SOURCE_USERS}
    ):
        raise ValueError("Expected a complete verified OpenPack 1.0.0 subject inventory")
    groups, splits = person_splits(list(SOURCE_USERS), seed=data_seed)
    (staging / "waveforms").mkdir(parents=True)
    (staging / "timestamps").mkdir()
    records = []
    for entry in sorted(entries, key=lambda e: e["name"]):
        path = source / entry["name"]
        if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"Corrupt OpenPack archive: {path}")
        user = path.stem
        with zipfile.ZipFile(path) as archive:
            selected: dict[str, dict[str, str]] = {}
            for member in archive.infolist():
                name = Path(member.filename)
                if name.is_absolute() or ".." in name.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe OpenPack ZIP member: {member.filename}")
                match = re.fullmatch(
                    r"(?:.*/)?(U0[12]\d{2})/(atr/atr01|annotation/openpack-operations-1hz|annotation/openpack-operations)/(S\d{4})\.csv",
                    member.filename,
                )
                if match:
                    owner, stream, session = match.groups()
                    if owner != user:
                        raise ValueError("Archive contains another user's recording")
                    kind = {
                        "atr/atr01": "imu",
                        "annotation/openpack-operations-1hz": "labels",
                        "annotation/openpack-operations": "intervals",
                    }[stream]
                    record = selected.setdefault(session, {})
                    if kind in record:
                        raise ValueError("Duplicate source stream")
                    record[kind] = member.filename
            if not selected or any(set(p) != {"imu", "labels", "intervals"} for p in selected.values()):
                raise ValueError(f"Incomplete native signal/operation inventory: {user}")
            for session, paths in sorted(selected.items()):
                records.append(
                    _prepare_record(
                        archive,
                        paths,
                        user=user,
                        session=session,
                        group=groups[user],
                        split=splits[user],
                        staging=staging,
                    )
                )
        logger.info("OpenPack prepared user=%s records=%d", user, len(selected))
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        summary[split] = {
            "groups": len({r["group_id"] for r in subset}),
            "records": len(subset),
            "signal_seconds": sum(r["num_samples"] / 30 for r in subset),
            "labeled_seconds": sum(r["labeled_seconds"] for r in subset),
            "class_seconds": [sum(r["class_seconds"][c] for r in subset) for c in range(10)],
        }
    manifest = {
        "format_version": 1,
        "dataset": "OPENPACK",
        "source_version": "1.0.0",
        "source_revision": SOURCE_REVISION,
        "source_url": provenance.get("source_url", "https://zenodo.org/records/8145223"),
        "source_provider": provenance.get("provider", "zenodo"),
        "license": "CC-BY-NC-SA-4.0",
        "protocol": "openpack_linked_person_10_3_3_v1",
        "scoring_protocol": "openpack_1hz",
        "target_coordinates": "sample",
        "data_seed": data_seed,
        "sampling_rate_hz": 30,
        "channel_names": ["acc_x", "acc_y", "acc_z"],
        "class_names": CLASS_NAMES,
        "normalization": "timestamp_regularize_polyphase30_then_clip3g_01_v1",
        "timing_correction_policy": "pinned_toolkit_first_row_and_U0108_initial_gaps_v1",
        "annotation_policy": "native_1hz_floor_second_null_mask_v1",
        "source_archives": entries,
        "identity_groups": groups,
        "summary": summary,
        "records": records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    staging.rename(output)
    logger.info("OpenPack preparation complete: %s", summary)
    return manifest
