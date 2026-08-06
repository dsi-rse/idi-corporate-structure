"""Tests for processor.failures — CorporateStructureFailureClassifier."""

import pytest

from idi_corporate_structure.failures import (
    CorporateStructureFailureClassifier,
    FailureType,
)


@pytest.fixture
def classifier() -> CorporateStructureFailureClassifier:
    """Return a CorporateStructureFailureClassifier instance."""
    return CorporateStructureFailureClassifier()


class TestCorporateStructureFailureClassifier:
    """Tests for CorporateStructureFailureClassifier.classify_from_response()."""

    def test_classify_rate_limit_response(self, classifier):
        response = {"status_code": 429}
        assert classifier.classify_from_response(response) == FailureType.RATE_LIMIT

    def test_classify_4xx_error(self, classifier):
        response = {"status_code": 404}
        assert classifier.classify_from_response(response) == FailureType.API_ERROR

    def test_classify_5xx_error(self, classifier):
        response = {"status_code": 503}
        assert classifier.classify_from_response(response) == FailureType.API_ERROR

    def test_classify_when_error_key_present(self, classifier):
        response = {"error": "Connection refused", "status_code": None}
        assert classifier.classify_from_response(response) == FailureType.API_ERROR

    def test_classify_when_no_status_code(self, classifier):
        response = {}
        assert classifier.classify_from_response(response) == FailureType.API_ERROR

    def test_accepts_kwargs_without_error(self, classifier):
        """classify_from_response accepts **kwargs per the interface."""
        response = {"status_code": 429}
        result = classifier.classify_from_response(response, extra="ignored")
        assert result == FailureType.RATE_LIMIT


class TestIsRetryable:
    """Tests for CorporateStructureFailureClassifier.is_retryable()."""

    @pytest.mark.parametrize(
        "failure_type",
        [
            FailureType.MISMATCHED_LENGTHS,
            FailureType.NO_FORM_DATA,
            FailureType.NO_10K_FILINGS,
            FailureType.NO_FILING_DIRECTORY,
            FailureType.NO_EXHIBIT_CONTENT,
            FailureType.MISSING_PERIOD_OF_REPORT,
        ],
    )
    def test_non_retryable_types(self, classifier, failure_type):
        assert not classifier.is_retryable(failure_type)

    @pytest.mark.parametrize(
        "failure_type",
        [
            FailureType.EXTRACTION_FAILED,
            FailureType.API_ERROR,
            FailureType.RATE_LIMIT,
        ],
    )
    def test_retryable_types(self, classifier, failure_type):
        assert classifier.is_retryable(failure_type)
