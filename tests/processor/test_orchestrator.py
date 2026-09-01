"""Tests for orchestrator argument parsing."""

import argparse
import datetime

import pandas as pd
import pytest

from idi_corporate_structure.orchestrator import get_args, get_dates, read_ciks_file

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

    def test_verbose_defaults_false_and_checkpoint_every_default(self, monkeypatch):
        monkeypatch.setattr("sys.argv", _FULL_ARGS)

        args = get_args()

        assert args.verbose is False
        assert args.checkpoint_every == 500

    def test_verbose_flag_enables_debug(self, monkeypatch):
        monkeypatch.setattr("sys.argv", [*_FULL_ARGS, "--verbose"])

        args = get_args()

        assert args.verbose is True


_BASE_ARGS = [a for a in _FULL_ARGS if a != "--daily"]


class TestCiksOverride:
    """Tests for the --ciks-override mode: file parsing and arg validation."""

    @pytest.fixture
    def ciks_file(self, tmp_path):
        path = tmp_path / "ciks.txt"
        path.write_text(
            "# CIKs of interest\n"
            "0000034088\n"
            "\n"
            "93410  # Chevron, trailing comment\n"
            "0000034088\n"  # duplicate of the first entry
        )
        return str(path)

    def test_read_ciks_file_normalizes_and_dedupes(self, ciks_file):
        assert read_ciks_file(ciks_file) == ("34088", "93410")

    def test_read_ciks_file_rejects_invalid_cik(self, tmp_path):
        path = tmp_path / "ciks.txt"
        path.write_text("not-a-cik\n")

        with pytest.raises(argparse.ArgumentTypeError, match="line 1"):
            read_ciks_file(path)

    def test_read_ciks_file_rejects_missing_file(self, tmp_path):
        with pytest.raises(argparse.ArgumentTypeError, match="Cannot read"):
            read_ciks_file(str(tmp_path / "does-not-exist.txt"))

    def test_parses_without_date_range(self, monkeypatch, ciks_file):
        monkeypatch.setattr("sys.argv", [*_BASE_ARGS, "--ciks-override", ciks_file])

        args = get_args()

        assert args.ciks_override == ("34088", "93410")
        assert args.start_date is None
        assert args.end_date is None
        assert args.daily is False

    @pytest.mark.parametrize(
        "extra",
        [
            ["--daily"],
            ["--start-date", "2024-01-01"],
            ["--end-date", "2024-01-15"],
            ["--look-back", "7"],
        ],
    )
    def test_conflicts_with_date_modes(self, monkeypatch, ciks_file, extra):
        monkeypatch.setattr("sys.argv", [*_BASE_ARGS, "--ciks-override", ciks_file, *extra])

        with pytest.raises(SystemExit):
            get_args()

    def test_empty_ciks_file_errors(self, monkeypatch, tmp_path):
        path = tmp_path / "ciks.txt"
        path.write_text("# only comments\n\n")
        monkeypatch.setattr("sys.argv", [*_BASE_ARGS, "--ciks-override", str(path)])

        with pytest.raises(SystemExit):
            get_args()


class TestGetDates:
    """Tests for get_dates() date-range resolution."""

    def test_explicit_mode_returns_given_dates(self):
        args = argparse.Namespace(
            daily=False,
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
        )

        start_date, end_date = get_dates(args)

        assert start_date == datetime.date(2024, 1, 1)
        assert end_date == datetime.date(2024, 1, 15)

    def test_daily_mode_derives_window_from_scraped_date(self, mocker):
        """Daily mode uses the manifest's most recent date_scraped, not filing_date."""
        manifest_df = pd.DataFrame(
            {
                "filing_date": ["2024-01-01", "2024-01-02"],
                "date_scraped": pd.to_datetime(
                    ["2024-06-09T12:00:00Z", "2024-06-10T08:00:00Z"], utc=True
                ),
            }
        )
        mocker.patch(
            "idi_corporate_structure.orchestrator.pd.read_parquet",
            return_value=manifest_df,
        )
        args = argparse.Namespace(
            daily=True,
            look_back=7,
            sec_bucket_prefix="test-bucket/sec",
            start_date=None,
            end_date=None,
        )

        start_date, end_date = get_dates(args)

        assert end_date == datetime.date(2024, 6, 10)
        assert start_date == datetime.date(2024, 6, 3)

    def test_daily_mode_raises_when_no_scraped_dates(self, mocker):
        manifest_df = pd.DataFrame({"date_scraped": pd.to_datetime([pd.NaT], utc=True)})
        mocker.patch(
            "idi_corporate_structure.orchestrator.pd.read_parquet",
            return_value=manifest_df,
        )
        args = argparse.Namespace(
            daily=True,
            look_back=7,
            sec_bucket_prefix="test-bucket/sec",
            start_date=None,
            end_date=None,
        )

        with pytest.raises(ValueError, match="date_scraped"):
            get_dates(args)
