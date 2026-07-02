"""Tests for orchestrator argument parsing."""

import pytest

from idi_corporate_structure.orchestrator import get_args

_FULL_ARGS = [
    "orchestrator.py",
    "--output-file",
    "out.parquet",
    "--failure-file",
    "fail.json",
    "--sec-bucket-prefix",
    "test-bucket/sec",
    "--openai-api-key",
    "fake-key",
    "--sec-user-agent",
    "Test test@test.com",
    "--daily",
]


class TestGetArgs:
    """Tests for get_args() required-argument validation."""

    def test_requires_sec_bucket_prefix(self, monkeypatch):
        """Omitting --sec-bucket-prefix must fail with a clean usage error.

        It previously wasn't marked required, so omitting it produced a raw
        AttributeError deep in main()/get_dates() instead.
        """
        args_without_prefix = [
            a for a in _FULL_ARGS if a not in ("--sec-bucket-prefix", "test-bucket/sec")
        ]
        monkeypatch.setattr("sys.argv", args_without_prefix)

        with pytest.raises(SystemExit):
            get_args()

    def test_succeeds_with_all_required_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", _FULL_ARGS)

        args = get_args()

        assert args.sec_bucket_prefix == "test-bucket/sec"
