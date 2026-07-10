"""Tests for processor.types — dataclasses and PipelineStats thread safety."""

import datetime
import threading

from idi_corporate_structure.types import (
    CompanyMeta,
    Filing,
    PipelineConfig,
    PipelineStats,
    Subsidiary,
)

_START_DATE = datetime.date(2024, 1, 1)
_END_DATE = datetime.date(2024, 1, 2)


class TestPipelineStats:
    """Tests for PipelineStats thread-safe counters."""

    def test_increment_default_by_one(self):
        stats = PipelineStats()
        stats.increment("total_filing")
        assert stats.total_filing == 1

    def test_increment_by_n(self):
        stats = PipelineStats()
        stats.increment("total_subsidiaries", 5)
        assert stats.total_subsidiaries == 5

    def test_increment_multiple_fields(self):
        stats = PipelineStats()
        stats.increment("total_filing")
        stats.increment("failed_filings", 3)
        stats.increment("failed_subsidiaries", 2)

        assert stats.total_filing == 1
        assert stats.failed_filings == 3
        assert stats.failed_subsidiaries == 2

    def test_thread_safe_concurrent_increments(self):
        """Many threads incrementing the same field should produce an exact count."""
        stats = PipelineStats()
        n_threads = 20
        increments_per_thread = 100

        def increment_many() -> None:
            for _ in range(increments_per_thread):
                stats.increment("total_filing")

        threads = [threading.Thread(target=increment_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_filing == n_threads * increments_per_thread

    def test_starts_at_zero(self):
        stats = PipelineStats()
        assert stats.total_filing == 0
        assert stats.failed_filings == 0
        assert stats.total_subsidiaries == 0
        assert stats.failed_subsidiaries == 0
        assert stats.skipped_filings == 0


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_valid_config_local_files(self, tmp_path):
        config = PipelineConfig(
            failure_file=str(tmp_path / "failures.json"),
            output_file=str(tmp_path / "subsidiaries.parquet"),
            start_date=_START_DATE,
            end_date=_END_DATE,
            sec_bucket="test-bucket",
        )

        assert config.sec_bucket == "test-bucket"
        assert config.num_workers == 10  # default
        assert config.rate_limit == 0.2  # default

    def test_creates_failure_directory_if_missing(self, tmp_path):
        failure_file = tmp_path / "nonexistent_dir" / "failures.json"
        PipelineConfig(
            failure_file=str(failure_file),
            output_file=str(tmp_path / "subsidiaries.parquet"),
            start_date=_START_DATE,
            end_date=_END_DATE,
            sec_bucket="test-bucket",
        )

        assert failure_file.parent.exists()

    def test_creates_output_directory_if_missing(self, tmp_path):
        output_file = tmp_path / "new_dir" / "subsidiaries.parquet"
        PipelineConfig(
            failure_file=str(tmp_path / "failures.json"),
            output_file=str(output_file),
            start_date=_START_DATE,
            end_date=_END_DATE,
            sec_bucket="test-bucket",
        )

        assert output_file.parent.exists()

    def test_skips_validation_for_s3_paths(self, tmp_path):
        """S3 paths should not trigger local directory creation."""
        config = PipelineConfig(
            failure_file="s3://my-bucket/failures.json",
            output_file="s3://my-bucket/subsidiaries.parquet",
            start_date=_START_DATE,
            end_date=_END_DATE,
            sec_bucket="test-bucket",
        )

        assert config.output_file == "s3://my-bucket/subsidiaries.parquet"
        assert not (tmp_path / "my-bucket").exists()

    def test_custom_num_workers(self, tmp_path):
        config = PipelineConfig(
            failure_file=str(tmp_path / "failures.json"),
            output_file=str(tmp_path / "subsidiaries.parquet"),
            start_date=_START_DATE,
            end_date=_END_DATE,
            sec_bucket="test-bucket",
            num_workers=4,
        )
        assert config.num_workers == 4


class TestFilingDataclass:
    """Tests for the Filing dataclass."""

    def test_filing_fields(self, sample_filing):
        assert sample_filing.cik == "0000320193"
        assert sample_filing.form_type == "10-K"
        assert "aapl-20240928.htm" in sample_filing.primary_document
        assert sample_filing.company_name == "APPLE INC"
        assert sample_filing.company.state_of_incorporation == "CA"

    def test_filing_equality(self):
        f1 = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            primary_document="",
        )
        f2 = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            primary_document="",
        )
        assert f1 == f2

    def test_filing_company_defaults_to_empty_company_meta(self):
        filing = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            primary_document="",
        )
        assert filing.company == CompanyMeta()

    def test_filing_exhibit_documents_defaults_to_empty_tuple(self):
        filing = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            primary_document="",
        )
        assert filing.exhibit_documents == ()


class TestFilingExhibitType:
    """Tests for Filing.exhibit_type computed property."""

    def _make_filing(self, form_type: str) -> Filing:
        return Filing(
            cik="",
            filing_date="",
            form_type=form_type,
            accession_number="",
            primary_document="",
        )

    # 10-K variants → Exhibit 21
    def test_exhibit_type_is_21_for_10k(self):
        assert self._make_filing("10-K").exhibit_type == "21"

    def test_exhibit_type_is_21_for_10k_slash_a(self):
        assert self._make_filing("10-K/A").exhibit_type == "21"

    def test_exhibit_type_is_21_for_10ksb(self):
        assert self._make_filing("10-KSB").exhibit_type == "21"

    def test_exhibit_type_is_21_for_10k_no_dash(self):
        assert self._make_filing("10K").exhibit_type == "21"

    # 20-F variants → Exhibit 8
    def test_exhibit_type_is_8_for_20f(self):
        assert self._make_filing("20-F").exhibit_type == "8"

    def test_exhibit_type_is_8_for_20f_slash_a(self):
        assert self._make_filing("20-F/A").exhibit_type == "8"

    def test_exhibit_type_is_8_for_20f_no_dash(self):
        assert self._make_filing("20F").exhibit_type == "8"


class TestSubsidiaryDataclass:
    """Tests for the Subsidiary dataclass."""

    def test_subsidiary_fields(self):
        sub = Subsidiary(
            parent_cik="0000320193",
            name="Apple Operations International",
            location="Ireland",
            filing_date="2024-09-28",
            form_type="10-K",
            exhibit_type="21",
            accession_number="0000320193-24-000123",
            exhibit_url="https://www.sec.gov/Archives/edgar/data/320193/ex21.htm",
        )
        assert sub.parent_cik == "0000320193"
        assert sub.name == "Apple Operations International"
        assert sub.location == "Ireland"
