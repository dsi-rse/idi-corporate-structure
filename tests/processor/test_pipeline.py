"""Tests for processor.pipeline — SubsidiaryPipeline."""

import queue
import threading
from pathlib import Path

import pandas as pd
import pytest
from idi_ftm2j_shared.sec import ScrapedDocument, ScrapedFiling

from idi_corporate_structure.extractor import (
    DocumentError,
    ExtractionTimeoutError,
    ExtractionTruncatedError,
)
from idi_corporate_structure.failures import FailureType
from idi_corporate_structure.pipeline import CompanyMetaFetchError
from idi_corporate_structure.types import CompanyMeta, Filing, Subsidiary
from tests.conftest import make_exhibit_response

# ── Data helpers ──────────────────────────────────────────────────────────────


def make_scraped_document(
    filename: str = "ex21.htm",
    doc_type: str = "EX-21.1",
    s3_key: str = "s3://test-bucket/sec/2024-01-01/10-K/0000320193/000032019324000123/ex21.htm",
) -> ScrapedDocument:
    """Build a minimal ScrapedDocument as would appear in a manifest.json."""
    return ScrapedDocument(
        filename=filename,
        url=f"https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/{filename}",
        type=doc_type,
        s3_key=s3_key,
    )


def make_scraped_filing(
    cik: str = "0000320193",
    accession_number: str = "0000320193-24-000123",
    form_type: str = "10-K",
    filing_date: str = "2024-01-01",
    company_name: str = "APPLE INC",
    documents: list | None = None,
) -> ScrapedFiling:
    """Build a minimal ScrapedFiling manifest for load_input tests."""
    return ScrapedFiling(
        cik=cik,
        accession_number=accession_number,
        form_type=form_type,
        filing_date=filing_date,
        index_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/index.htm",
        company_name=company_name,
        documents=documents if documents is not None else [make_scraped_document()],
    )


def make_subsidiary(
    parent_cik: str = "0000320193",
    name: str = "Test Sub LLC",
    location: str = "Delaware",
    accession_number: str = "0000320193-24-000123",
) -> Subsidiary:
    """Build a minimal Subsidiary for process/extract-worker tests."""
    return Subsidiary(
        parent_cik=parent_cik,
        filing_date="2024-09-28",
        form_type="10-K",
        exhibit_type="21",
        accession_number=accession_number,
        exhibit_url="https://example.com/ex21.htm",
        name=name,
        location=location,
    )


def make_filing(cik: str = "0000320193", accession_number: str = "0000320193-24-000123") -> Filing:
    """Build a minimal Filing not tied to the sample_filing fixture."""
    return Filing(
        cik=cik,
        filing_date="2024-09-28",
        form_type="10-K",
        accession_number=accession_number,
        primary_document="",
    )


# ── _select_exhibit_documents ────────────────────────────────────────────────


class TestSelectExhibitDocuments:
    """Tests for SubsidiaryPipeline._select_exhibit_documents()."""

    def test_matches_ex21_for_10k(self, pipeline):
        scraped = make_scraped_filing(
            documents=[
                make_scraped_document(doc_type="EX-21.1"),
                make_scraped_document(filename="ex10.htm", doc_type="EX-10.1"),
            ]
        )

        result = pipeline._select_exhibit_documents(scraped, "21")

        assert len(result) == 1
        assert result[0].type == "EX-21.1"

    def test_matches_ex8_for_20f(self, pipeline):
        scraped = make_scraped_filing(
            documents=[make_scraped_document(filename="ex8.htm", doc_type="EX-8.1")]
        )

        result = pipeline._select_exhibit_documents(scraped, "8")

        assert len(result) == 1

    def test_returns_empty_tuple_when_no_match(self, pipeline):
        scraped = make_scraped_filing(
            documents=[make_scraped_document(filename="ex10.htm", doc_type="EX-10.1")]
        )

        result = pipeline._select_exhibit_documents(scraped, "21")

        assert result == ()

    def test_case_and_punctuation_insensitive(self, pipeline):
        scraped = make_scraped_filing(documents=[make_scraped_document(doc_type="ex-21")])

        result = pipeline._select_exhibit_documents(scraped, "21")

        assert len(result) == 1


# ── _should_skip ──────────────────────────────────────────────────────────────


class TestShouldSkip:
    """Tests for SubsidiaryPipeline._should_skip()."""

    def test_false_when_not_processed_or_failed(self, pipeline, sample_filing):
        assert pipeline._should_skip(sample_filing, processed_accessions=set()) is False

    def test_true_when_accession_already_processed(self, pipeline, sample_filing):
        processed = {sample_filing.accession_number}
        assert pipeline._should_skip(sample_filing, processed_accessions=processed) is True

    def test_true_when_in_failure_registry(self, pipeline, sample_filing):
        pipeline.failure_registry._entries.add((sample_filing.cik, sample_filing.accession_number))

        assert pipeline._should_skip(sample_filing, processed_accessions=set()) is True


# ── _load_processed_accessions ───────────────────────────────────────────────


class TestLoadProcessedAccessions:
    """Tests for SubsidiaryPipeline._load_processed_accessions()."""

    def test_returns_empty_set_when_no_output_file(self, pipeline):
        assert pipeline._load_processed_accessions() == set()

    def test_returns_accession_numbers_from_existing_output(self, pipeline):
        pd.DataFrame({"accession_number": ["ACC1", "ACC2", "ACC1"]}).to_parquet(
            pipeline.config.output_file
        )

        assert pipeline._load_processed_accessions() == {"ACC1", "ACC2"}


# ── _fetch_company_meta ───────────────────────────────────────────────────────


class TestFetchCompanyMeta:
    """Tests for SubsidiaryPipeline._fetch_company_meta()."""

    def test_parses_business_address_and_state_of_incorporation(self, pipeline):
        pipeline.sec_client.query_endpoint.return_value = {
            "data": {
                "stateOfIncorporation": "DE",
                "addresses": {"business": {"street1": "1 Infinite Loop", "city": "Cupertino"}},
                "tickers": ["AAPL"],
                "exchanges": ["Nasdaq"],
            }
        }

        meta = pipeline._fetch_company_meta("320193")

        assert meta.state_of_incorporation == "DE"
        assert meta.business_street1 == "1 Infinite Loop"
        assert meta.business_city == "Cupertino"
        assert meta.tickers == ("AAPL",)
        assert meta.exchanges == ("Nasdaq",)

    def test_maps_business_state_from_state_or_country(self, pipeline):
        """business_state comes from the SEC ``stateOrCountry`` key (not ``stateOrCounty``)."""
        pipeline.sec_client.query_endpoint.return_value = {
            "data": {"addresses": {"business": {"stateOrCountry": "CA"}}}
        }

        meta = pipeline._fetch_company_meta("320193")

        assert meta.business_state == "CA"

    def test_defaults_to_blank_fields_when_data_missing(self, pipeline):
        pipeline.sec_client.query_endpoint.return_value = {}

        meta = pipeline._fetch_company_meta("320193")

        assert meta == CompanyMeta()

    def test_raises_on_sec_request_error(self, pipeline):
        """A failed SEC request must raise, not silently yield a blank CompanyMeta."""
        pipeline.sec_client.query_endpoint.return_value = {"error": "Timeout querying SEC"}

        with pytest.raises(CompanyMetaFetchError):
            pipeline._fetch_company_meta("320193")

    def test_does_not_cache_on_sec_request_error(self, pipeline):
        """After a failed request, a later call re-queries instead of returning a cached blank."""
        pipeline.sec_client.query_endpoint.side_effect = [
            {"error": "Timeout querying SEC"},
            {"data": {"stateOfIncorporation": "DE"}},
        ]

        with pytest.raises(CompanyMetaFetchError):
            pipeline._fetch_company_meta("320193")
        meta = pipeline._fetch_company_meta("320193")

        assert meta.state_of_incorporation == "DE"
        assert pipeline.sec_client.query_endpoint.call_count == 2

    def test_zero_pads_cik_in_request_url(self, pipeline):
        pipeline.sec_client.query_endpoint.return_value = {}

        pipeline._fetch_company_meta("320193")

        called_url = pipeline.sec_client.query_endpoint.call_args.kwargs["sec_url"]
        assert "CIK0000320193.json" in called_url

    def test_null_exchange_entry_becomes_empty_string(self, pipeline):
        """Regression test: SEC returns exchanges: [null] for some unlisted tickers.

        A raw None element used to survive into CompanyMeta.exchanges and crash
        extractor.py's ",".join(filing.company.exchanges) downstream.
        """
        pipeline.sec_client.query_endpoint.return_value = {
            "data": {"tickers": ["VRXA"], "exchanges": [None]}
        }

        meta = pipeline._fetch_company_meta("2079109")

        assert meta.exchanges == ("",)
        assert ",".join(meta.exchanges) == ""

    def test_null_ticker_entry_becomes_empty_string(self, pipeline):
        pipeline.sec_client.query_endpoint.return_value = {
            "data": {"tickers": [None, "AAPL"], "exchanges": ["", "Nasdaq"]}
        }

        meta = pipeline._fetch_company_meta("320193")

        assert meta.tickers == ("", "AAPL")
        assert ",".join(meta.tickers) == ",AAPL"

    def test_caches_result_per_cik(self, pipeline):
        """A CIK with multiple filings should only trigger one request.

        Not one SEC submissions-JSON request per filing.
        """
        pipeline.sec_client.query_endpoint.return_value = {"data": {"stateOfIncorporation": "DE"}}

        first = pipeline._fetch_company_meta("320193")
        second = pipeline._fetch_company_meta("320193")

        assert first == second
        assert pipeline.sec_client.query_endpoint.call_count == 1

    def test_does_not_cache_across_different_ciks(self, pipeline):
        pipeline.sec_client.query_endpoint.return_value = {"data": {"stateOfIncorporation": "DE"}}

        pipeline._fetch_company_meta("320193")
        pipeline._fetch_company_meta("1067491")

        assert pipeline.sec_client.query_endpoint.call_count == 2


# ── load_input ────────────────────────────────────────────────────────────────


class TestLoadInput:
    """Tests for SubsidiaryPipeline.load_input()."""

    def test_returns_filings_with_matching_exhibits(self, pipeline, mocker):
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing()],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        filings = pipeline.load_input()

        assert len(filings) == 1
        assert filings[0].cik == "0000320193"
        assert len(filings[0].exhibit_documents) == 1

    def test_sets_total_documents_from_selected_exhibits(self, pipeline, mocker):
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[
                make_scraped_filing(),
                make_scraped_filing(accession_number="0000320193-24-000999"),
            ],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        filings = pipeline.load_input()

        assert pipeline._total_documents == sum(len(f.exhibit_documents) for f in filings)
        assert pipeline._total_documents == 2

    def test_increments_total_filing_per_scraped_filing(self, pipeline, mocker):
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing(), make_scraped_filing(accession_number="ACC2")],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        pipeline.load_input()

        assert pipeline.stats.total_filing == 2

    def test_excludes_and_records_failure_for_filing_with_no_exhibits(self, pipeline, mocker):
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing(documents=[])],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        filings = pipeline.load_input()

        assert filings == []
        assert pipeline.stats.failed_filings == 1
        assert ("0000320193", "0000320193-24-000123") in pipeline.failure_registry

    def test_skips_already_processed_accession(self, pipeline, mocker):
        pd.DataFrame({"accession_number": ["0000320193-24-000123"]}).to_parquet(
            pipeline.config.output_file
        )
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing()],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        filings = pipeline.load_input()

        assert filings == []
        assert pipeline.stats.skipped_filings == 1

    def test_skips_and_records_retryable_failure_on_meta_error(self, pipeline, mocker):
        """A metadata fetch error skips the filing and records a retryable failure.

        The failure must not be persisted to the registry, so the filing is
        reprocessed on the next run rather than being marked processed with blank
        parent metadata.
        """
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing()],
        )
        mocker.patch.object(
            pipeline,
            "_fetch_company_meta",
            side_effect=CompanyMetaFetchError("0000320193", "url"),
        )

        filings = pipeline.load_input()

        assert filings == []
        assert pipeline.stats.failed_filings == 1
        assert ("0000320193", "0000320193-24-000123") not in pipeline.failure_registry

    def test_queries_filings_by_scraped_date(self, pipeline, mocker):
        """load_input must pull filings by scraped_date, not filing_date."""
        mock_iter = mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[make_scraped_filing()],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        pipeline.load_input()

        assert mock_iter.call_args.kwargs["search_by"] == "scraped_date"
        assert "include_failures" not in mock_iter.call_args.kwargs

    def test_respects_input_sample_size(self, pipeline, mocker):
        mocker.patch.object(pipeline, "_INPUT_SAMPLE_SIZE", 1)
        mocker.patch(
            "idi_corporate_structure.pipeline.iter_filings_by_form_type",
            return_value=[
                make_scraped_filing(accession_number="ACC1"),
                make_scraped_filing(accession_number="ACC2"),
            ],
        )
        mocker.patch.object(pipeline, "_fetch_company_meta", return_value=CompanyMeta())

        filings = pipeline.load_input()

        assert len(filings) == 1


# ── _record_failure ───────────────────────────────────────────────────────────


class TestRecordFailure:
    """Tests for SubsidiaryPipeline._record_failure()."""

    def test_logs_at_given_level(self, pipeline, mocker):
        mock_warn = mocker.patch.object(pipeline.logger, "warning")

        pipeline._record_failure(
            ("CIK1", "ACC1"), FailureType.NO_SUBSIDIARIES, "warning", "no subs: %s", "ACC1"
        )

        mock_warn.assert_called_once_with("no subs: %s", "ACC1")

    def test_increments_default_stat_key(self, pipeline):
        pipeline._record_failure(("CIK1", "ACC1"), FailureType.EXTRACTION_FAILED, "error", "msg")

        assert pipeline.stats.failed_subsidiaries == 1

    def test_increments_custom_stat_keys(self, pipeline):
        pipeline._record_failure(
            ("CIK1", "ACC1"),
            FailureType.TRUNCATED_ERROR,
            "error",
            "msg",
            stat_keys=("failed_subsidiaries", "truncated_extractions"),
        )

        assert pipeline.stats.failed_subsidiaries == 1
        assert pipeline.stats.truncated_extractions == 1

    def test_adds_non_retryable_failure_to_registry(self, pipeline):
        pipeline._record_failure(("CIK1", "ACC1"), FailureType.NO_EXHIBIT_FOUND, "warning", "msg")

        assert ("CIK1", "ACC1") in pipeline.failure_registry

    def test_does_not_persist_retryable_failure_to_registry(self, pipeline):
        """EXTRACTION_FAILED is retryable, so it should not be persisted."""
        pipeline._record_failure(("CIK1", "ACC1"), FailureType.EXTRACTION_FAILED, "error", "msg")

        assert ("CIK1", "ACC1") not in pipeline.failure_registry


# ── _report_extraction ────────────────────────────────────────────────────────


class TestReportExtraction:
    """Tests for SubsidiaryPipeline._report_extraction()."""

    def test_increments_total_subsidiaries(self, pipeline, sample_filing):
        pipeline._report_extraction(
            num_chunks=1,
            ungrounded_name=0,
            ungrounded_location=0,
            num_subsidiaries=3,
            filing=sample_filing,
        )

        assert pipeline.stats.total_subsidiaries == 3

    def test_increments_chunked_extractions_when_multiple_chunks(self, pipeline, sample_filing):
        pipeline._report_extraction(
            num_chunks=5,
            ungrounded_name=0,
            ungrounded_location=0,
            num_subsidiaries=1,
            filing=sample_filing,
        )

        assert pipeline.stats.chunked_extractions == 1

    def test_does_not_increment_chunked_extractions_for_single_chunk(self, pipeline, sample_filing):
        pipeline._report_extraction(
            num_chunks=1,
            ungrounded_name=0,
            ungrounded_location=0,
            num_subsidiaries=1,
            filing=sample_filing,
        )

        assert pipeline.stats.chunked_extractions == 0

    def test_increments_ungrounded_counts(self, pipeline, sample_filing):
        pipeline._report_extraction(
            num_chunks=1,
            ungrounded_name=2,
            ungrounded_location=4,
            num_subsidiaries=1,
            filing=sample_filing,
        )

        assert pipeline.stats.ungrounded_name == 2
        assert pipeline.stats.ungrounded_location == 4

    def test_records_no_subsidiaries_failure_when_zero(self, pipeline, sample_filing):
        pipeline._report_extraction(
            num_chunks=1,
            ungrounded_name=0,
            ungrounded_location=0,
            num_subsidiaries=0,
            filing=sample_filing,
        )

        assert pipeline.stats.zero_subsidiaries == 1
        assert (sample_filing.cik, sample_filing.accession_number) in pipeline.failure_registry


# ── _extract_worker ───────────────────────────────────────────────────────────


class TestExtractWorker:
    """Tests for SubsidiaryPipeline._extract_worker()."""

    def _start_worker(self, pipeline, work_queue, subsidiaries):
        threading.Thread(
            target=pipeline._extract_worker,
            args=(work_queue, subsidiaries),
            daemon=True,
        ).start()

    def test_calls_extractor_with_filing_and_contents(self, pipeline, sample_filing):
        exhibit = make_exhibit_response()
        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, exhibit))
        work_queue.join()

        pipeline.extractor.extract.assert_called_once_with(sample_filing, exhibit)

    def test_appends_batch_to_subsidiaries_list(self, pipeline, sample_filing):
        subsidiary = make_subsidiary(parent_cik=sample_filing.cik)
        pipeline.extractor.extract.return_value = ([subsidiary], 0, 0, 1)

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert subsidiaries == [subsidiary]

    def test_marks_work_task_done_on_success(self, pipeline, sample_filing):
        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()  # completes only if task_done() was called

    def test_marks_work_task_done_on_exception(self, pipeline, sample_filing):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()  # completes only if task_done() is called in finally

    def test_increments_failed_subsidiaries_on_exception(self, pipeline, sample_filing):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.failed_subsidiaries == 1

    def test_records_extraction_failed_on_generic_exception(self, pipeline, sample_filing, mocker):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")
        spy = mocker.spy(pipeline.failure_registry, "add")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        spy.assert_called_once_with(
            (sample_filing.cik, sample_filing.accession_number), FailureType.EXTRACTION_FAILED
        )

    def test_generic_exception_logs_traceback_and_exhibit_url(
        self, pipeline, sample_filing, mocker
    ):
        """Regression test: the error and exhibit URL must be diagnosable from the log line."""
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")
        exception_spy = mocker.spy(pipeline.logger, "exception")
        exhibit = make_exhibit_response()

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, exhibit))
        work_queue.join()

        exception_spy.assert_called_once()
        logged_args = exception_spy.call_args.args
        assert "GPT error" in str(logged_args)
        assert exhibit["url"] in logged_args

    def test_records_document_error(self, pipeline, sample_filing, mocker):
        pipeline.extractor.extract.side_effect = DocumentError("too long")
        spy = mocker.spy(pipeline.failure_registry, "add")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        spy.assert_called_once_with(
            (sample_filing.cik, sample_filing.accession_number), FailureType.DOCUMENT_ERROR
        )

    def test_timeout_error_increments_timeout_and_failed(self, pipeline, sample_filing):
        pipeline.extractor.extract.side_effect = ExtractionTimeoutError("timed out")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.timeout_subsidiaries == 1
        assert pipeline.stats.failed_subsidiaries == 1

    def test_chunked_extraction_increments_stat(self, pipeline, sample_filing):
        pipeline.extractor.extract.return_value = ([], 0, 0, 5)  # 5 chunks

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.chunked_extractions == 1

    def test_one_shot_does_not_increment_chunked(self, pipeline, sample_filing):
        pipeline.extractor.extract.return_value = ([], 0, 0, 1)  # single chunk

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.chunked_extractions == 0

    def test_truncated_extraction_increments_truncated_and_failed(
        self, pipeline, sample_filing, mocker
    ):
        pipeline.extractor.extract.side_effect = ExtractionTruncatedError("output cut off")
        spy = mocker.spy(pipeline.failure_registry, "add")

        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.truncated_extractions == 1
        assert pipeline.stats.failed_subsidiaries == 1
        spy.assert_called_once_with(
            (sample_filing.cik, sample_filing.accession_number), FailureType.TRUNCATED_ERROR
        )

    def test_increments_extracted_documents_per_item(self, pipeline, sample_filing):
        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.extracted_documents == 1

    def test_checkpoints_at_configured_cadence(self, pipeline, sample_filing, mocker):
        pipeline.config.checkpoint_every = 1
        checkpointed = threading.Event()
        spy = mocker.patch.object(
            pipeline, "_checkpoint_results", side_effect=lambda *_: checkpointed.set()
        )
        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))

        assert checkpointed.wait(timeout=2)
        spy.assert_called_with(subsidiaries)

    def test_does_not_checkpoint_when_disabled(self, pipeline, sample_filing, mocker):
        pipeline.config.checkpoint_every = 0
        spy = mocker.patch.object(pipeline, "_checkpoint_results")
        work_queue, subsidiaries = queue.Queue(), []
        self._start_worker(pipeline, work_queue, subsidiaries)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        spy.assert_not_called()


# ── _checkpoint_results ───────────────────────────────────────────────────────


class TestCheckpointResults:
    """Tests for SubsidiaryPipeline._checkpoint_results()."""

    def test_writes_snapshot_to_output_file(self, pipeline):
        pipeline._checkpoint_results([make_subsidiary()])

        df = pd.read_parquet(pipeline.config.output_file)
        assert len(df) == 1

    def test_skips_when_checkpoint_already_in_progress(self, pipeline, mocker):
        save = mocker.patch.object(pipeline, "save_output")
        # Simulate an in-flight checkpoint holding the lock: the non-blocking
        # acquire must fail and the call must return without writing.
        pipeline._checkpoint_lock.acquire()
        try:
            pipeline._checkpoint_results([make_subsidiary()])
        finally:
            pipeline._checkpoint_lock.release()

        save.assert_not_called()


# ── _extract_pdf_text ─────────────────────────────────────────────────────────


class TestExtractPdfText:
    """Tests for SubsidiaryPipeline._extract_pdf_text()."""

    def _mock_pdf(self, mocker, page_texts: list[str]):
        pages = []
        for text in page_texts:
            page = mocker.MagicMock()
            page.extract_text.return_value = text
            pages.append(page)
        mock_pdf = mocker.MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = mocker.MagicMock(return_value=False)
        mock_pdf.pages = pages
        return mocker.patch(
            "idi_corporate_structure.pipeline.pdfplumber.open", return_value=mock_pdf
        )

    def test_extracts_text_from_pdf(self, pipeline, sample_filing, mocker):
        self._mock_pdf(mocker, ["Subsidiary A — Delaware"])

        result = pipeline._extract_pdf_text(
            b"%PDF content", "https://example.com/ex21.pdf", sample_filing
        )

        assert result == "Subsidiary A — Delaware"

    def test_joins_multiple_pages_with_double_newline(self, pipeline, sample_filing, mocker):
        self._mock_pdf(mocker, ["page one", "page two"])

        result = pipeline._extract_pdf_text(
            b"%PDF content", "https://example.com/ex21.pdf", sample_filing
        )

        assert result == "page one\n\npage two"

    def test_returns_empty_string_and_records_failure_on_parse_error(
        self, pipeline, sample_filing, mocker
    ):
        """Regression test: a parse error must not raise UnboundLocalError."""
        mocker.patch(
            "idi_corporate_structure.pipeline.pdfplumber.open",
            side_effect=Exception("corrupt PDF"),
        )

        result = pipeline._extract_pdf_text(
            b"garbage", "https://example.com/ex21.pdf", sample_filing
        )

        assert result == ""
        assert pipeline.stats.failed_subsidiaries == 1


# ── _fetch_exhibit ────────────────────────────────────────────────────────────


class TestFetchExhibit:
    """Tests for SubsidiaryPipeline._fetch_exhibit()."""

    def test_returns_empty_list_when_no_exhibit_documents(self, pipeline, sample_filing):
        sample_filing.exhibit_documents = ()

        assert pipeline._fetch_exhibit(sample_filing) == []

    def test_skips_documents_without_s3_key(self, pipeline, sample_filing, mocker):
        sample_filing.exhibit_documents = (make_scraped_document(s3_key=""),)
        mock_load = mocker.patch("idi_corporate_structure.pipeline.load_content")

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []
        mock_load.assert_not_called()

    def test_records_failure_when_content_missing(self, pipeline, sample_filing, mocker):
        sample_filing.exhibit_documents = (make_scraped_document(),)
        mocker.patch("idi_corporate_structure.pipeline.load_content", return_value=b"")

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []
        assert pipeline.stats.failed_subsidiaries == 1

    def test_records_failure_and_continues_when_s3_read_raises(
        self, pipeline, sample_filing, mocker
    ):
        """Regression test: an S3 error (e.g. NoSuchBucket) must not crash the pipeline."""
        sample_filing.exhibit_documents = (
            make_scraped_document(filename="ex21.htm"),
            make_scraped_document(filename="ex21b.htm"),
        )
        mocker.patch(
            "idi_corporate_structure.pipeline.load_content",
            side_effect=[Exception("NoSuchBucket"), b"Apple Operations LLC"],
        )

        result = pipeline._fetch_exhibit(sample_filing)

        assert len(result) == 1
        assert result[0]["data"] == "Apple Operations LLC"
        assert pipeline.stats.failed_subsidiaries == 1

    def test_decodes_htm_as_html_to_text(self, pipeline, sample_filing, mocker):
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.htm"),)
        mocker.patch(
            "idi_corporate_structure.pipeline.load_content",
            return_value=b"<html><body>Apple Operations LLC (Delaware)</body></html>",
        )

        result = pipeline._fetch_exhibit(sample_filing)

        assert len(result) == 1
        assert "<html>" not in result[0]["data"]
        assert "Apple Operations LLC" in result[0]["data"]

    def test_decodes_txt_as_plain_text(self, pipeline, sample_filing, mocker):
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.txt"),)
        mocker.patch(
            "idi_corporate_structure.pipeline.load_content",
            return_value=b"Apple Operations LLC",
        )

        result = pipeline._fetch_exhibit(sample_filing)

        assert result[0]["data"] == "Apple Operations LLC"

    def test_extracts_pdf_via_extract_pdf_text(self, pipeline, sample_filing, mocker):
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.pdf"),)
        mocker.patch("idi_corporate_structure.pipeline.load_content", return_value=b"%PDF")
        mocker.patch.object(pipeline, "_extract_pdf_text", return_value="extracted text")

        result = pipeline._fetch_exhibit(sample_filing)

        assert result[0]["data"] == "extracted text"

    def test_does_not_enqueue_empty_pdf_text(self, pipeline, sample_filing, mocker):
        """A PDF that extracts to empty text is not enqueued (would burn an OpenAI call)."""
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.pdf"),)
        mocker.patch("idi_corporate_structure.pipeline.load_content", return_value=b"%PDF")
        mocker.patch.object(pipeline, "_extract_pdf_text", return_value="")

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []

    def test_records_failure_for_empty_non_pdf_text(self, pipeline, sample_filing, mocker):
        """HTML that renders to only whitespace is skipped and recorded as NO_EXHIBIT_CONTENT."""
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.htm"),)
        mocker.patch(
            "idi_corporate_structure.pipeline.load_content",
            return_value=b"<html><body>   </body></html>",
        )

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []
        assert pipeline.stats.failed_subsidiaries == 1

    @pytest.mark.parametrize(
        "filename,stat_key",
        [
            ("ex21.htm", "htm_exhibits"),
            ("ex21.html", "html_exhibits"),
            ("ex21.txt", "txt_exhibits"),
            ("ex21.pdf", "pdf_exhibits"),
        ],
    )
    def test_increments_type_counter(self, pipeline, sample_filing, mocker, filename, stat_key):
        sample_filing.exhibit_documents = (make_scraped_document(filename=filename),)
        mocker.patch("idi_corporate_structure.pipeline.load_content", return_value=b"content")
        mocker.patch.object(pipeline, "_extract_pdf_text", return_value="text")

        pipeline._fetch_exhibit(sample_filing)

        assert getattr(pipeline.stats, stat_key) == 1

    def test_unsupported_extension_does_not_crash_or_increment_counters(
        self, pipeline, sample_filing, mocker
    ):
        """Regression test: an oddball extension must not AttributeError on stats.increment."""
        sample_filing.exhibit_documents = (make_scraped_document(filename="ex21.xml"),)
        mocker.patch("idi_corporate_structure.pipeline.load_content", return_value=b"<xml/>")

        result = pipeline._fetch_exhibit(sample_filing)

        assert result[0]["data"] == "<xml/>"
        for stat_key in ("htm_exhibits", "html_exhibits", "txt_exhibits", "pdf_exhibits"):
            assert getattr(pipeline.stats, stat_key) == 0


# ── process ───────────────────────────────────────────────────────────────────


class TestProcess:
    """Tests for SubsidiaryPipeline.process()."""

    def test_returns_subsidiaries_from_extractor(self, pipeline, mocker):
        filings = [make_filing(cik=f"CIK{i}", accession_number=f"ACC{i}") for i in range(3)]
        mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[make_exhibit_response()])
        pipeline.extractor.extract.side_effect = [
            ([make_subsidiary(parent_cik=f.cik)], 0, 0, 1) for f in filings
        ]

        results = pipeline.process(filings)

        assert len(results) == 3
        assert all(isinstance(r, Subsidiary) for r in results)

    def test_returns_empty_list_for_empty_input(self, pipeline):
        assert pipeline.process([]) == []

    def test_calls_fetch_exhibit_for_each_filing(self, pipeline, mocker):
        filings = [make_filing(cik=f"CIK{i}", accession_number=f"ACC{i}") for i in range(4)]
        mock_fetch = mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[])

        pipeline.process(filings)

        assert mock_fetch.call_count == 4

    def test_increments_queued_documents_per_exhibit(self, pipeline, mocker):
        filing = make_filing()
        mocker.patch.object(
            pipeline,
            "_fetch_exhibit",
            return_value=[make_exhibit_response(), make_exhibit_response()],
        )

        pipeline.process([filing])

        assert pipeline.stats.queued_documents == 2

    def test_handles_extractor_exception_gracefully(self, pipeline, mocker):
        """A failed extraction should not crash the pipeline — other filings still processed."""
        filings = [make_filing(cik=f"CIK{i}", accession_number=f"ACC{i}") for i in range(3)]
        mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[make_exhibit_response()])

        subsidiary = make_subsidiary(parent_cik="CIK_OK")
        pipeline.extractor.extract.side_effect = [
            RuntimeError("GPT error"),
            ([subsidiary], 0, 0, 1),
            ([subsidiary], 0, 0, 1),
        ]

        results = pipeline.process(filings)

        # One failure + two successes
        assert len(results) == 2
        assert pipeline.stats.failed_subsidiaries >= 1


# ── save_output ───────────────────────────────────────────────────────────────


class TestSaveOutput:
    """Tests for SubsidiaryPipeline.save_output()."""

    def _make_subsidiary(self, name: str, accession: str = "0000320193-24-000123") -> Subsidiary:
        return Subsidiary(
            parent_cik="0000320193",
            parent_name="APPLE INC",
            parent_state_of_incorporation="CA",
            name=name,
            location="Ireland",
            filing_date="2024-09-28",
            form_type="10-K",
            exhibit_type="21",
            accession_number=accession,
            exhibit_url="https://www.sec.gov/Archives/edgar/data/320193/ex21.htm",
        )

    def test_writes_parquet_file(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Operations International")]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 1

    def test_output_contains_all_subsidiary_fields(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Sales International")]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert result_df.iloc[0]["name"] == "Apple Sales International"
        assert result_df.iloc[0]["parent_cik"] == "0000320193"
        assert result_df.iloc[0]["location"] == "Ireland"

    def test_adds_date_added_column(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Operations International")]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert "date_added" in result_df.columns
        assert result_df.iloc[0]["date_added"] is not None

    def test_deduplicates_within_filing(self, pipeline):
        """Same parent_cik + accession_number + name should be written once."""
        subsidiaries = [
            self._make_subsidiary("Apple Operations International"),
            self._make_subsidiary("Apple Operations International"),
        ]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 1

    def test_keeps_same_name_across_different_filings(self, pipeline):
        """Same subsidiary name in two different filings should produce two rows."""
        subsidiaries = [
            self._make_subsidiary(
                "Apple Operations International", accession="0000320193-23-000001"
            ),
            self._make_subsidiary(
                "Apple Operations International", accession="0000320193-24-000002"
            ),
        ]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 2

    def test_normalizes_location_strings_on_write(self, pipeline):
        """Footnote markers, sentinels, and SEC state codes should be canonicalized."""
        subs = [
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_state_of_incorporation="DE",
                name="Acme China Sub",
                location="PRC",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_state_of_incorporation="E9",
                name="Acme Mexico Sub",
                location="Mexico(2)",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_state_of_incorporation="L2",
                name="Acme Mystery Sub",
                location="Unknown",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
        ]

        pipeline.save_output(subs)

        result_df = pd.read_parquet(pipeline.config.output_file).set_index("name")
        assert result_df.loc["Acme China Sub", "location"] == "China"
        assert result_df.loc["Acme China Sub", "parent_state_of_incorporation"] == "Delaware"
        assert result_df.loc["Acme Mexico Sub", "location"] == "Mexico"
        assert result_df.loc["Acme Mexico Sub", "parent_state_of_incorporation"] == "Cayman Islands"
        assert result_df.loc["Acme Mystery Sub", "location"] == ""
        assert result_df.loc["Acme Mystery Sub", "parent_state_of_incorporation"] == "Ireland"

    def test_empty_list_does_not_raise_when_no_existing_file(self, pipeline):
        """save_output([]) must not raise when no output file exists yet.

        pd.DataFrame([]) has zero columns (not just zero rows), so indexing
        combined_subsidiaries_df["location"] used to raise KeyError when
        process() extracted nothing and no output file existed yet.
        """
        pipeline.save_output([])

        assert not Path(pipeline.config.output_file).exists()

    def test_empty_list_leaves_existing_file_untouched(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Operations International")]
        pipeline.save_output(subsidiaries)

        pipeline.save_output([])

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 1
        assert result_df.iloc[0]["name"] == "Apple Operations International"


# ── display_stats ─────────────────────────────────────────────────────────────


class TestDisplayStats:
    """Tests for SubsidiaryPipeline.display_stats()."""

    def test_logs_filing_counts(self, pipeline, mocker):
        pipeline.stats.increment("total_filing", 10)
        pipeline.stats.increment("failed_filings", 2)
        mock_logger = mocker.patch.object(pipeline, "logger")

        pipeline.display_stats()

        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "10" in logged
        assert "2" in logged

    def test_logs_subsidiary_counts(self, pipeline, mocker):
        pipeline.stats.increment("total_subsidiaries", 50)
        pipeline.stats.increment("failed_subsidiaries", 3)
        mock_logger = mocker.patch.object(pipeline, "logger")

        pipeline.display_stats()

        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "50" in logged
        assert "3" in logged

    def test_logs_section_headers(self, pipeline, mocker):
        mock_logger = mocker.patch.object(pipeline, "logger")

        pipeline.display_stats()

        logged_args = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Filings" in arg for arg in logged_args)
        assert any("Subsidiaries" in arg for arg in logged_args)
        assert any("Exhibits by type" in arg for arg in logged_args)
        assert any("=" in arg for arg in logged_args)


# ── run (early exit) ──────────────────────────────────────────────────────────


class TestRunEarlyExit:
    """Tests for Pipeline.run() early-exit when load_input returns nothing."""

    def test_skips_process_when_nothing_to_process(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_process = mocker.patch.object(pipeline, "process")

        pipeline.run()

        mock_process.assert_not_called()

    def test_skips_save_output_when_nothing_to_process(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_save = mocker.patch.object(pipeline, "save_output")

        pipeline.run()

        mock_save.assert_not_called()

    def test_flushes_failure_registry_on_early_exit(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_flush = mocker.patch.object(pipeline.failure_registry, "flush")

        pipeline.run()

        mock_flush.assert_called_once()
