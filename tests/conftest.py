"""Shared fixtures for the test suite."""

import datetime
from unittest.mock import MagicMock

import pytest
from idi_ftm2j_shared.api import SecClient

from idi_corporate_structure.extractor import GptExtractor
from idi_corporate_structure.pipeline import SubsidiaryPipeline
from idi_corporate_structure.types import CompanyMeta, Filing, PipelineConfig

# ── Data helpers ──────────────────────────────────────────────────────────────


def make_exhibit_response(
    content: str = (
        "<html><body>\nApple Operations LLC (Delaware)\nApple Europe Ltd (Ireland)\n</body></html>"
    ),
) -> dict:
    """Build a minimal exhibit document dict as passed to GptExtractor.extract()."""
    return {
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/ex21.htm",
        "data": content,
    }


# ── Core fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_20f_filing() -> Filing:
    """A realistic 20-F Filing dataclass instance (Coincheck Group)."""
    return Filing(
        cik="0001913847",
        filing_date="2025-07-30",
        form_type="20-F",
        accession_number="0001628280-25-036727",
        primary_document=(
            "https://www.sec.gov/Archives/edgar/data/1913847/000162828025036727/index.htm"
        ),
        company_name="Coincheck Group",
    )


@pytest.fixture
def sample_filing() -> Filing:
    """A realistic Filing dataclass instance."""
    return Filing(
        cik="0000320193",
        filing_date="2024-09-28",
        form_type="10-K",
        accession_number="0000320193-24-000123",
        primary_document=(
            "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324000123"
            "/aapl-20240928.htm"
        ),
        company_name="APPLE INC",
        company=CompanyMeta(state_of_incorporation="CA"),
    )


@pytest.fixture
def mock_sec_client() -> MagicMock:
    """A MagicMock that satisfies the SecClient interface."""
    client = MagicMock(spec=SecClient)
    client.SEC_HEADERS = {"User-Agent": "test test@test.com"}
    client.SEC_URL = "https://www.sec.gov/Archives/edgar/data"
    client.rate_limit.return_value = None
    return client


@pytest.fixture
def mock_extractor() -> MagicMock:
    """A MagicMock GptExtractor that returns an empty subsidiary list by default."""
    extractor = MagicMock(spec=GptExtractor)
    extractor.extract.return_value = ([], 0, 0, 1)
    return extractor


@pytest.fixture
def pipeline(tmp_path, mock_sec_client, mock_extractor) -> SubsidiaryPipeline:
    """A SubsidiaryPipeline wired with a temp failure/output path and mocked dependencies."""
    config = PipelineConfig(
        failure_file=str(tmp_path / "failures.json"),
        output_file=str(tmp_path / "subsidiaries.parquet"),
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 2),
        sec_bucket="test-bucket",
        rate_limit=0.0,
        num_workers=2,
        failure_flush_every=100,
    )

    return SubsidiaryPipeline(
        config=config,
        sec_client=mock_sec_client,
        extractor=mock_extractor,
    )
