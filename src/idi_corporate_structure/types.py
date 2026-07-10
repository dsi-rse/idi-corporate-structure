"""Data types for the corporate structure pipeline."""

# Standard application imports
import datetime
import pathlib
import re
import threading
from dataclasses import dataclass, field

# Third party applications
from idi_ftm2j_shared.sec import ScrapedDocument

_REMOTE_SCHEMES = ("s3://", "https://", "http://", "gs://")
SUPPORTED_EXHIBIT_EXTENSIONS = frozenset({"HTM", "HTML", "TXT", "PDF"})

# Single source of truth for which filings carry a subsidiaries exhibit
# (Exhibit 21 domestic / Exhibit 8 foreign)
TARGET_FORM_TYPES = [
    # Domestic — Exhibit 21
    "10-K",
    "10-K/A",
    "10-KT",
    "10-KT/A",
    # Foreign — Exhibit 8
    "20-F",
    "20-F/A",
    "20FR12B",
    "20FR12B/A",
    "20FR12G",
    "20FR12G/A",
]


def _is_local(path: str) -> bool:
    """Return True if the path refers to a local filesystem location.

    A path is considered remote when it begins with one of the known URI
    schemes in ``_REMOTE_SCHEMES`` (``s3://``, ``https://``, ``http://``,
    ``gs://``).

    Args:
        path: File path or URI string to test.

    Returns:
        True if the path does not start with a known remote scheme,
        False otherwise.
    """
    return not path.startswith(_REMOTE_SCHEMES)


@dataclass(frozen=True)
class CompanyMeta:
    """Per-CIK company metadata from the SEC submissions JSON.

    Fetched once per CIK and shared across all of that company's filings.
    Business address only — mailing address adds little for entity matching.
    """

    state_of_incorporation: str = ""
    business_street1: str = ""
    business_street2: str = ""
    business_city: str = ""
    business_state: str = ""  # addresses.business.stateOrCountry — "CA" or a country code
    business_zip: str = ""
    business_country: str = ""
    business_country_code: str = ""
    tickers: tuple[str, ...] = ()  # ("POWW",)
    exchanges: tuple[str, ...] = ()  # ("Nasdaq",)


@dataclass
class Filing:
    """Represents a single SEC 10-K filing with its metadata and document URLs."""

    cik: str
    filing_date: str
    form_type: str
    accession_number: str
    primary_document: str
    company_name: str = ""
    company: CompanyMeta = field(default_factory=CompanyMeta)
    exhibit_documents: tuple[ScrapedDocument, ...] = ()  # EX-21/EX-8

    @property
    def exhibit_type(self) -> str:
        """Return the exhibit number for the filing's subsidiary list.

        10-K filers use Exhibit 21; 20-F filers use Exhibit 8.
        """
        return "8" if re.match(r"20-?F", self.form_type) else "21"


@dataclass
class PipelineConfig:
    """Configuration for the subsidiary pipeline."""

    failure_file: str
    output_file: str
    start_date: datetime.date
    end_date: datetime.date
    sec_bucket: str
    openai_api_key: str = ""
    failure_flush_every: int = 50
    rate_limit: float = 0.2
    num_workers: int = 10

    def __post_init__(self) -> None:
        """Validate existence of local files."""
        if _is_local(self.failure_file) and not pathlib.Path(self.failure_file).parent.exists():
            pathlib.Path(self.failure_file).parent.mkdir(parents=True, exist_ok=True)
        if _is_local(self.output_file) and not pathlib.Path(self.output_file).parent.exists():
            pathlib.Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineStats:
    """Thread-safe counters tracking pipeline progress and failures."""

    total_filing: int = 0
    failed_filings: int = 0
    skipped_filings: int = 0
    total_subsidiaries: int = 0
    failed_subsidiaries: int = 0
    timeout_subsidiaries: int = 0
    truncated_extractions: int = 0
    chunked_extractions: int = 0
    zero_subsidiaries: int = 0
    ungrounded_name: int = 0
    ungrounded_location: int = 0
    dropped_subsidiaries: int = 0
    htm_exhibits: int = 0
    html_exhibits: int = 0
    txt_exhibits: int = 0
    pdf_exhibits: int = 0
    queued_documents: int = 0
    extracted_documents: int = 0

    def __post_init__(self) -> None:
        """Initialize the pipeline stats."""
        self._lock = threading.Lock()

    def increment(self, field: str, n: int = 1) -> None:
        """Increment the pipeline stats by a given amount.

        Args:
            field: The field to increment
            n: The amount to increment the field by
        """
        with self._lock:
            setattr(self, field, getattr(self, field) + n)


@dataclass
class Subsidiary:
    """A single subsidiary entity extracted from an Exhibit 21 document."""

    parent_cik: str
    filing_date: str
    form_type: str
    exhibit_type: str
    accession_number: str
    exhibit_url: str
    name: str
    location: str
    parent_name: str = ""
    parent_state_of_incorporation: str = ""
    parent_business_street1: str = ""
    parent_business_street2: str = ""
    parent_business_city: str = ""
    parent_business_state: str = ""
    parent_business_zip: str = ""
    parent_business_country: str = ""
    parent_business_country_code: str = ""
    parent_tickers: str = ""
    parent_exchanges: str = ""
    source_quote: str = ""
