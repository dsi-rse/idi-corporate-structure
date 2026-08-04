"""Tests for common.failures — FailureRegistry."""

import json
import threading

from idi_ftm2j_shared.failures import FailureRegistry

from idi_corporate_structure.failures import (
    CorporateStructureFailureClassifier,
    FailureType,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_registry(tmp_path, flush_every: int = 10) -> FailureRegistry:
    """Build a FailureRegistry backed by a temp file."""
    failure_file = str(tmp_path / "failures.json")
    return FailureRegistry(
        file_path=failure_file,
        classifier=CorporateStructureFailureClassifier(),
        flush_every=flush_every,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestFailureRegistryAdd:
    """Tests for FailureRegistry.add()."""

    def test_non_retryable_failure_is_added(self, tmp_path):
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.MISMATCHED_LENGTHS)

        assert ("0001234567", "0001234567-24-000001") in registry

    def test_retryable_failure_is_skipped(self, tmp_path):
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.EXTRACTION_FAILED)

        assert ("0001234567", "0001234567-24-000001") not in registry

    def test_duplicate_add_is_idempotent(self, tmp_path):
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.NO_FORM_DATA)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.NO_FORM_DATA)

        # Still only one entry
        assert len(registry._entries) == 1

    def test_all_non_retryable_types_are_added(self, tmp_path):
        non_retryable = [
            FailureType.MISMATCHED_LENGTHS,
            FailureType.NO_FORM_DATA,
            FailureType.NO_10K_FILINGS,
            FailureType.NO_FILING_DIRECTORY,
            FailureType.NO_EXHIBIT_CONTENT,
        ]
        registry = make_registry(tmp_path)
        for i, ft in enumerate(non_retryable):
            registry.add((f"CIK{i:010d}", f"ACC{i:020d}"), ft)

        assert len(registry._entries) == len(non_retryable)

    def test_all_retryable_types_are_skipped(self, tmp_path):
        retryable = [
            FailureType.EXTRACTION_FAILED,
            FailureType.API_ERROR,
            FailureType.RATE_LIMIT,
        ]
        registry = make_registry(tmp_path)
        for i, ft in enumerate(retryable):
            registry.add((f"CIK{i:010d}", f"ACC{i:020d}"), ft)

        assert len(registry._entries) == 0


class TestFailureRegistryPersistence:
    """Tests for FailureRegistry load/save persistence."""

    def test_flush_writes_entries_to_disk(self, tmp_path):
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.MISMATCHED_LENGTHS)
        registry.flush()

        failure_file = tmp_path / "failures.json"
        assert failure_file.exists()
        data = json.loads(failure_file.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0] == ["0001234567", "0001234567-24-000001"]

    def test_load_restores_entries_from_disk(self, tmp_path):
        # Write initial registry and flush
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.NO_FORM_DATA)
        registry.flush()

        # New registry instance reads from the same file
        registry2 = make_registry(tmp_path)
        assert ("0001234567", "0001234567-24-000001") in registry2

    def test_auto_flush_at_threshold(self, tmp_path):
        registry = make_registry(tmp_path, flush_every=3)

        registry.add(("CIK0000000001", "ACC0000000001"), FailureType.MISMATCHED_LENGTHS)
        registry.add(("CIK0000000002", "ACC0000000002"), FailureType.NO_FORM_DATA)

        failure_file = tmp_path / "failures.json"
        assert not failure_file.exists()  # not flushed yet

        registry.add(("CIK0000000003", "ACC0000000003"), FailureType.NO_10K_FILINGS)

        # Third add triggers flush
        assert failure_file.exists()
        data = json.loads(failure_file.read_text())
        assert len(data["entries"]) == 3

    def test_load_from_missing_file_starts_empty(self, tmp_path):
        failure_file = str(tmp_path / "nonexistent.json")
        registry = FailureRegistry(
            file_path=failure_file,
            classifier=CorporateStructureFailureClassifier(),
        )
        assert len(registry._entries) == 0

    def test_load_from_corrupt_file_starts_empty(self, tmp_path):
        failure_file = tmp_path / "corrupt.json"
        failure_file.write_text("not valid json{{{{")

        registry = FailureRegistry(
            file_path=str(failure_file),
            classifier=CorporateStructureFailureClassifier(),
        )
        assert len(registry._entries) == 0

    def test_reason_is_stored(self, tmp_path):
        registry = make_registry(tmp_path)
        registry.add(("0001234567", "0001234567-24-000001"), FailureType.MISMATCHED_LENGTHS)
        registry.flush()

        data = json.loads((tmp_path / "failures.json").read_text())
        reason = data["reasons"]["0001234567 0001234567-24-000001"]
        assert reason == str(FailureType.MISMATCHED_LENGTHS)


class TestFailureRegistryThreadSafety:
    """Tests for FailureRegistry concurrent access."""

    def test_concurrent_adds_do_not_corrupt_entries(self, tmp_path):
        """Multiple threads adding distinct entries should all be recorded."""
        registry = make_registry(tmp_path, flush_every=1000)
        errors = []

        def add_entries(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    registry.add((f"CIK{i:010d}", f"ACC{i:020d}"), FailureType.MISMATCHED_LENGTHS)
            except Exception as e:  # noqa: BLE001 — record any thread error for the assertion
                errors.append(e)

        threads = [threading.Thread(target=add_entries, args=(i * 50, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(registry._entries) == 200
