"""Synthetic OpenPack source integrity, clocks, channels and linked-person checks."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from trd.data.openpack import (
    PERSON_ALIASES,
    SOURCE_USERS,
    _prepare_record,
    file_digest,
    operation_index,
    person_splits,
    prepare_openpack,
    resample_acceleration,
)

pytestmark = pytest.mark.unit


def make_source(root: Path, *, missing: bool = False, unsafe: bool = False) -> Path:
    """Write tiny archives for the real identity inventory, using synthetic motion."""
    root.mkdir()
    entries = []
    origin = 1609459200000
    for user in SOURCE_USERS:
        path = root / f"{user}.zip"
        session = "S0500"
        imu = "unixtime,acc_x,acc_y,acc_z,gyro_x,box,id\n" + "".join(
            f"{origin + i * 30},-3,0,3,999999,SECRET,900\n" for i in range(101)
        )
        labels = "unixtime,user,session,box,id\n" + "".join(
            f"{origin + i * 1000},{user},{session},metadata,{label}\n"
            for i, label in enumerate([100, 8100, 1000, 8200])
        )
        intervals = (
            "uuid,user,session,box,id,operation,start,end,actions\n"
            f"example,{user},{session},metadata,100,Picking,"
            "2021-01-01T09:00:00+09:00,2021-01-01T09:00:01+09:00,[]\n"
        )
        with zipfile.ZipFile(path, "w") as archive:
            prefix = f"openpack/v1.0.0/{user}"
            archive.writestr(f"{prefix}/atr/atr01/{session}.csv", imu)
            archive.writestr(f"{prefix}/annotation/openpack-operations-1hz/{session}.csv", labels)
            if not (missing and user == SOURCE_USERS[0]):
                archive.writestr(f"{prefix}/annotation/openpack-operations/{session}.csv", intervals)
            if unsafe and user == SOURCE_USERS[0]:
                archive.writestr("../escape", "invalid")
        entries.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "md5": file_digest(path, "md5"),
                "sha256": file_digest(path),
            }
        )
    (root / "source_manifest.json").write_text(json.dumps({"release": "1.0.0", "archives": entries}))
    return root


def test_linked_people_and_development_exposure_are_preserved():
    groups, splits = person_splits(list(SOURCE_USERS), seed=20260913)
    assert len(set(groups.values())) == 16
    for alias, person in PERSON_ALIASES.items():
        assert groups[alias] == groups[person] == person
        assert splits[alias] == splits[person]
    assert splits["U0209"] == "train"
    assert [len({groups[u] for u in groups if splits[u] == s}) for s in ("train", "val", "test")] == [10, 3, 3]
    assert (groups, splits) == person_splits(list(reversed(SOURCE_USERS)), seed=20260913)


def test_prepared_manifest_retains_the_actual_source_provider(tmp_path):
    source = make_source(tmp_path / "source")
    path = source / "source_manifest.json"
    provenance = json.loads(path.read_text())
    provenance.update(provider="google_drive_selective", source_url="https://drive.google.com/drive/folders/fixture")
    path.write_text(json.dumps(provenance))
    prepared = prepare_openpack(source, tmp_path / "prepared")
    assert prepared["source_provider"] == "google_drive_selective"
    assert prepared["source_url"] == provenance["source_url"]


def test_native_clock_controls_resampling_and_metadata_never_becomes_signal(tmp_path):
    source = make_source(tmp_path / "source")
    result = prepare_openpack(source, tmp_path / "prepared")
    record = next(r for r in result["records"] if r["user_id"] == "U0209")
    values = np.load(tmp_path / "prepared" / record["path"], allow_pickle=False)
    assert values.shape == (91, 3)  # 101 native rows at 30 ms span three seconds.
    np.testing.assert_allclose(values, np.broadcast_to([0, 0.5, 1], values.shape), atol=1e-3)
    assert values.dtype == np.float32
    assert record["signal_audit"]["native_period_ms"] == 30
    assert record["annotations"][:3] == [[0, 1, 0], [1, 2, -1], [2, 3, 9]]
    assert record["source_operation_intervals"][0][0] == 1609459200000
    assert record["native_last_timestamps_ms"] == [1609459200990, 1609459201980, 1609459202970, 1609459203000]
    corrected = next(r for r in result["records"] if r["user_id"] == "U0101")
    assert corrected["origin_unix_ms"] == 1609459200030
    assert corrected["signal_audit"]["source_first_row_removed"]
    assert result == prepare_openpack(source, tmp_path / "again")
    with pytest.raises(FileExistsError):
        prepare_openpack(source, tmp_path / "prepared")


@pytest.mark.parametrize("problem,match", [("missing", "Incomplete"), ("unsafe", "Unsafe"), ("corrupt", "Corrupt")])
def test_bad_source_cannot_publish_prepared_data(tmp_path, problem, match):
    source = make_source(tmp_path / "source", missing=problem == "missing", unsafe=problem == "unsafe")
    if problem == "corrupt":
        with (source / "U0101.zip").open("ab") as handle:
            handle.write(b"corruption")
    with pytest.raises(ValueError, match=match):
        prepare_openpack(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    with pytest.raises(FileExistsError):
        prepare_openpack(source, tmp_path / "prepared")


@pytest.mark.parametrize("times", [[0, 30, 30], [60, 30, 0], [0, 30, 150]])
def test_duplicate_reversed_and_gapped_clocks_fail(times):
    with pytest.raises(ValueError, match="timestamps|gap"):
        resample_acceleration(np.array(times), np.zeros((3, 3)))


@pytest.mark.parametrize("matching_source", [False, True])
def test_pinned_initial_gap_correction_preserves_suffix_and_raw_label_tracks(tmp_path, monkeypatch, matching_source):
    origin = 1609459200000
    times = [origin + i * 30 + (1230 if i >= 244 else 0) + (5850 if i >= 245 else 0) for i in range(301)]
    imu = (
        "unixtime,acc_x,acc_y,acc_z\n" + "".join(f"{t},{99 if i < 245 else 0},0,0\n" for i, t in enumerate(times))
    ).encode()
    labels = (
        "unixtime,user,session,box,id\n" + "".join(f"{origin + i * 1000},U0108,S0400,box,100\n" for i in range(18))
    ).encode()
    intervals = (
        "uuid,user,session,box,id,operation,start,end,actions\n"
        "fixture,U0108,S0400,box,100,Picking,2021-01-01T00:00:00+00:00,2021-01-01T00:00:20+00:00,[]\n"
    ).encode()
    monkeypatch.setattr(
        "trd.data.openpack.U0108_S0400_IMU_SHA256", hashlib.sha256(imu).hexdigest() if matching_source else "0" * 64
    )
    (tmp_path / "waveforms").mkdir()
    (tmp_path / "timestamps").mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in [("imu", imu), ("labels", labels), ("intervals", intervals)]:
            archive.writestr(name, data)
    with zipfile.ZipFile(buffer) as archive:
        kwargs = dict(user="U0108", session="S0400", group="U0108", split="train", staging=tmp_path)
        paths = {k: k for k in ["imu", "labels", "intervals"]}
        if not matching_source:
            with pytest.raises(ValueError, match="audited exact source"):
                _prepare_record(archive, paths, **kwargs)
            assert not list((tmp_path / "waveforms").iterdir())
            return
        record = _prepare_record(archive, paths, **kwargs)
    assert record["origin_unix_ms"] == origin + 14430
    np.testing.assert_array_equal(np.load(tmp_path / record["native_timestamps_path"]), times[245:])
    np.testing.assert_allclose(np.load(tmp_path / record["path"]), 0.5, atol=1e-6)
    assert record["operation_labels_1hz"][0] == [origin, 0]
    assert record["source_operation_intervals"][0][:2] == [origin, origin + 20000]
    assert record["source_files"]["imu"] == hashlib.sha256(imu).hexdigest()
    audit = record["signal_audit"]
    assert audit["source_native_samples"] == 301 and audit["native_samples"] == 56
    assert audit["source_prefix_correction"]["removed_samples"] == 245
    assert audit["source_prefix_correction"]["removed_elapsed_ms"] == 14430
    assert audit["irregular_steps"] == 0


def test_picking_and_unknown_are_distinct_and_illegal_operations_fail():
    assert [operation_index(x) for x in [100, 1000, 8100, 8200]] == [0, 9, -1, -1]
    with pytest.raises(ValueError, match="Unknown"):
        operation_index(0)
