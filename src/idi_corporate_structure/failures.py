"""Corporate structure failure classification."""

# Standard library imports
from enum import StrEnum

# Third-party imports
from idi_ftm2j_shared.failures import FailureClassifier

_HTTP_RATE_LIMIT = 429
_HTTP_OK = 200
_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500


class FailureType(StrEnum):
    """Failure type for the corporate structure pipeline."""

    MISMATCHED_LENGTHS = "mismatched_lengths"  # Parallel filing arrays have unequal lengths
    NO_FORM_DATA = "no_form_data"  # Filing arrays have no data
    NO_10K_FILINGS = "no_10k_filings"  # CIK exists but has no 10-K forms
    NO_OVERFLOW_FILINGS = "no_overflow_filings"  # CIK exists but has no overflow filings
    NO_FILING_DIRECTORY = (
        "no_filing_directory"  # SEC queried filing but no directory listing was found
    )
    NO_EXHIBIT_CONTENT = "no_exhibit_content"  # Exhibit has no content
    NO_EXHIBIT_FOUND = "no_exhibit_found"  # No exhibit file found in filing directory
    DOCUMENT_ERROR = "document_error"  # Document is too long to process
    EXTRACTION_FAILED = "extraction_failed"  # GPT returned no structured data
    TIMEOUT_ERROR = "timeout_error"  # OpenAI API timed out
    TRUNCATED_ERROR = "truncated_error"  # Model response cut off at output token limit
    API_ERROR = "api_error"  # HTTP failure fetching filing document
    RATE_LIMIT = "rate_limit"  # SEC rate limit (429)
    NO_SUBSIDIARIES = "no_subsidiaries"  # No subsidiaries found for filing
    MISSING_REPORT_DATE = (
        "missing_report_date"  # Neither the manifest nor the submissions JSON dates the filing
    )


class CorporateStructureFailureClassifier(FailureClassifier):
    """Classifies failures for the corporate structure pipeline."""

    _DO_NOT_RETRY = frozenset(
        {
            FailureType.MISMATCHED_LENGTHS,
            FailureType.NO_FORM_DATA,
            FailureType.NO_10K_FILINGS,
            FailureType.NO_EXHIBIT_CONTENT,
            FailureType.NO_EXHIBIT_FOUND,
            FailureType.NO_FILING_DIRECTORY,
            FailureType.DOCUMENT_ERROR,
            FailureType.NO_OVERFLOW_FILINGS,
            FailureType.NO_SUBSIDIARIES,
            FailureType.MISSING_REPORT_DATE,
            FailureType.TRUNCATED_ERROR,
        }
    )

    @property
    def do_not_retry(self) -> frozenset:
        """Return the set of failure types that should not be retried."""
        return self._DO_NOT_RETRY

    def classify_from_response(self, response: dict, **kwargs) -> FailureType:
        """Classify failure from an HTTP response fetching a filing document.

        Args:
            response: Response dict with status_code and optional error.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The classified FailureType.
        """
        status_code = response.get("status_code")
        has_error = "error" in response

        if has_error or status_code is None:
            return FailureType.API_ERROR

        if status_code == _HTTP_RATE_LIMIT:
            return FailureType.RATE_LIMIT

        if _HTTP_CLIENT_ERROR_MIN <= status_code < _HTTP_SERVER_ERROR_MIN:
            return FailureType.API_ERROR

        return FailureType.API_ERROR
