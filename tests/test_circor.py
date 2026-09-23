"""Synthetic source-release checks for CirCor preparation and identity grouping."""

import csv
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest

from trd.data.circor import (
    extract_circor_archive,
    file_sha256,
    identity_groups,
    prepare_circor,
    read_circor_annotations,
)
from trd.data.temporal import Interval, validate_intervals

pytestmark = pytest.mark.unit


def make_source(root: Path) -> Path:
    """Build a tiny complete source release with a repeated identity."""
    root.mkdir()
    (root / "training_data").mkdir()
    ids = [str(v) for v in range(10, 18)]
    with (root / "training_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Patient ID", "Additional ID"])
        writer.writeheader()
        for patient in ids:
            writer.writerow({"Patient ID": patient, "Additional ID": {"10": "11", "11": "10"}.get(patient, "")})
    (root / "RECORDS").write_text("".join(f"training_data/{v}_AV\n" for v in ids))
    (root / "LICENSE.txt").write_text("Synthetic test fixture, not real patient data\n")
    for patient in ids:
        base = root / "training_data" / f"{patient}_AV"
        with wave.open(str(base.with_suffix(".wav")), "wb") as handle:
            handle.setparams((1, 2, 4000, 400, "NONE", "not compressed"))
            handle.writeframes((np.sin(np.arange(400) / 7) * 1000).astype("<i2").tobytes())
        base.with_suffix(".hea").write_text(f"{patient}_AV 1 4000 400\n")
        base.with_suffix(".tsv").write_text("0\t0.02\t1\n0.02\t0.04\t2\n0.04\t0.06\t0\n0.06\t0.08\t3\n0.08\t0.1\t4\n")
    refresh_checksums(root)
    return root


def refresh_checksums(root: Path) -> None:
    """Create checksums for synthetic files only."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(f"{file_sha256(p)}  {p.relative_to(root)}\n" for p in files))


def test_prepare_preserves_time_and_groups(tmp_path):
    source = make_source(tmp_path / "source")
    result = prepare_circor(source, tmp_path / "prepared", data_seed=12)
    rows = {r["patient_id"]: r for r in result["records"]}
    assert rows["10"]["group_id"] == rows["11"]["group_id"]
    assert rows["10"]["split"] == rows["11"]["split"]
    assert sum(v["groups"] for v in result["summary"].values()) == 7
    row = rows["10"]
    assert row["annotations"][2] == [0.04, 0.06, -1]
    assert row["labeled_seconds"] == pytest.approx(0.08)
    signal = np.load(tmp_path / "prepared" / row["path"])
    assert signal.shape == (400, 1)
    assert float(signal.mean()) == pytest.approx(0, abs=1e-6)
    assert float(signal.std()) == pytest.approx(1, abs=1e-6)
    again = prepare_circor(source, tmp_path / "second", data_seed=12)
    assert result == again


def test_source_corruption_and_missing_annotation_fail(tmp_path):
    source = make_source(tmp_path / "source")
    (source / "training_data/10_AV.tsv").unlink()
    with pytest.raises(ValueError, match="Missing or corrupt"):
        prepare_circor(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared/manifest.json").exists()


def test_invalid_intervals_and_existing_output_fail(tmp_path):
    source = make_source(tmp_path / "source")
    (source / "training_data/10_AV.tsv").write_text("0\t-0.06\t1\n")
    refresh_checksums(source)
    with pytest.raises(ValueError, match="Invalid CirCor annotation bounds"):
        prepare_circor(source, tmp_path / "prepared")
    with pytest.raises(FileExistsError):
        prepare_circor(source, tmp_path / "prepared")


def test_unresolved_identity_fails():
    with pytest.raises(ValueError, match="absent"):
        identity_groups([{"Patient ID": "1", "Additional ID": "2"}])


def test_archive_path_traversal_fails_before_extraction(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_circor_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


def test_source_mask_protocol_preserves_original_rows_and_event_onsets(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("0\t0\t28\n0\t0.06\t1\n0.04\t0.1001\t2\n")
    result = read_circor_annotations(path, duration=0.1)
    assert result.intervals == [Interval(0, 0.04, 0), Interval(0.04, 0.06, -1), Interval(0.06, 0.1, 1)]
    assert result.events[1].start == 0.04  # The masking boundary must not become a new reference onset.
    assert result.source_rows[0] == (0, 0, 28)
    assert result.source_rows[-1][1] == 0.1001
    assert result.audit["zero_duration_rows"] == result.audit["clipped_endpoints"] == 1
    assert result.audit["conflict_seconds"] == pytest.approx(0.02)
    assert not result.audit["quarantined"]
    with pytest.raises(ValueError, match="Overlapping"):
        validate_intervals([Interval(0, 0.06, 0), Interval(0.04, 0.1, 1)], duration=0.1, num_classes=4)


def test_quarantined_annotations_keep_waveform_and_identity(tmp_path):
    source = make_source(tmp_path / "source")
    (source / "training_data/10_AV.tsv").write_text("0\t0.02\t1\n0.02\t0.04\t2\n0\t0.01\t1\n")
    refresh_checksums(source)
    result = prepare_circor(source, tmp_path / "prepared")
    record = next(r for r in result["records"] if r["patient_id"] == "10")
    assert record["annotation_audit"]["quarantined"]
    assert record["annotations"] == [[0, 0.1, -1]]
    assert record["event_annotations"] == [] and record["labeled_seconds"] == 0
    assert np.load(tmp_path / "prepared" / record["path"]).shape == (400, 1)
    assert len(result["records"]) == 8


@pytest.mark.parametrize("text", ["0\t0.1\t28\n", "0\t0.1003\t1\n", "nan\t0.1\t1\n"])
def test_mask_policy_does_not_hide_invalid_positive_rows(tmp_path, text):
    path = tmp_path / "invalid.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match="Invalid CirCor"):
        read_circor_annotations(path, duration=0.1)


def test_duplicate_same_class_support_is_unioned_once(tmp_path):
    path = tmp_path / "duplicate.tsv"
    path.write_text("0\t0.08\t1\n0.02\t0.1\t1\n")
    result = read_circor_annotations(path, duration=0.1)
    assert result.intervals == result.events == [Interval(0, 0.1, 0)]
    assert result.audit["conflict_seconds"] == 0


def test_zero_duration_only_source_is_explicitly_unlabeled(tmp_path):
    path = tmp_path / "zero.tsv"
    path.write_text("0\t0\t28\n")
    result = read_circor_annotations(path, duration=1)
    assert result.intervals == [Interval(0, 1, -1)] and result.events == []
    assert result.audit["zero_duration_only"] and not result.audit["quarantined"]
    path.write_text("")
    with pytest.raises(ValueError, match="Empty annotation"):
        read_circor_annotations(path, duration=1)
