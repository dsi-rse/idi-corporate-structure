"""Pipeline for extracting subsidiary data from SEC 10-K Exhibit 21 filings."""

# Standard application imports
import dataclasses
import datetime
import io
import logging
import os
import queue
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import islice

# Third party imports
import pandas as pd
import pdfplumber
from idi_ftm2j_shared.api import SecClient
from idi_ftm2j_shared.failures import FailureRegistry
from idi_ftm2j_shared.logs import get_logger
from idi_ftm2j_shared.sec import ScrapedDocument, ScrapedFiling, iter_filings_by_form_type
from idi_ftm2j_shared.storage import load_content

# Application imports
from idi_corporate_structure.extractor import (
    DocumentError,
    ExtractionTimeoutError,
    ExtractionTruncatedError,
    GptExtractor,
    html_to_text,
)
from idi_corporate_structure.failures import (
    CorporateStructureFailureClassifier,
    FailureType,
)
from idi_corporate_structure.normalization import (
    normalize_parent_location,
    normalize_subsidiary_location,
)
from idi_corporate_structure.types import (
    TARGET_FORM_TYPES,
    CompanyMeta,
    Filing,
    PipelineConfig,
    PipelineStats,
    Subsidiary,
)


class CompanyMetaFetchError(Exception):
    """Raised when the SEC submissions request for a CIK fails.

    Signals a failed request (as opposed to a successful-but-sparse response) so
    callers can skip the filing and retry it on a later run instead of caching and
    persisting a blank ``CompanyMeta``.
    """


class Pipeline(ABC):
    """Baseline class for processing piplines."""

    def __init__(
        self,
        config: PipelineConfig,
        sec_client: SecClient,
        extractor: GptExtractor,
    ) -> None:
        """Initialize the pipeline with config, SEC client, and extractor.

        Args:
            config: Pipeline configuration including input/output paths and tuning
                parameters.
            sec_client: Configured SEC EDGAR API client used for fetching filings.
            extractor: Extractor instance responsible for parsing subsidiary data
                from exhibit documents.
        """
        self.config = config
        self.extractor = extractor
        self.sec_client = sec_client
        self.stats = PipelineStats()
        self.logger = get_logger(type(self).__name__)

    @abstractmethod
    def load_input(self) -> list:
        """Load input data and return a list of items to process.

        Returns:
            List of input items. The concrete element type is defined by each
            subclass (e.g. ``list[Filing]``).
        """
        ...

    @abstractmethod
    def process(self, input_list: list) -> list:
        """Process each item in the input list and return a list of results.

        Args:
            input_list: Items returned by :meth:`load_input`.

        Returns:
            List of processed results. The concrete element type is defined by
            each subclass (e.g. ``list[Subsidiary]``).
        """
        ...

    @abstractmethod
    def save_output(self, processed_list: list) -> None:
        """Persist the processed results to the configured output destination.

        Args:
            processed_list: Items returned by :meth:`process`.

        Returns:
            None
        """
        ...

    @abstractmethod
    def display_stats(self) -> None:
        """Log or display a summary of pipeline processing statistics.

        Returns:
            None
        """

    def run(self) -> None:
        """Execute the full pipeline: load → process → save → display stats.

        Calls :meth:`load_input`, :meth:`process`, :meth:`save_output`, and
        :meth:`display_stats` in sequence, then logs the total elapsed time.

        Returns:
            None
        """
        start_time = datetime.datetime.now(datetime.UTC)

        input_data = self.load_input()
        self.logger.info("Located %d filings with exhibits to process", len(input_data))

        if input_data:
            results = self.process(input_data)
            self.save_output(results)
            self.display_stats()
        else:
            self.logger.info("No input data found, skipping pipeline")

        end_time = datetime.datetime.now(datetime.UTC)
        self.logger.info("Elasped time: %s", end_time - start_time)


class _ResultSink:
    """Owns the extracted-results buffer and its checkpoint lifecycle.

    Encapsulates the mutable results list so it is never shared with or returned to
    the caller directly: the results worker feeds it via :meth:`add` / :meth:`checkpoint`,
    and :meth:`process` collects the unflushed remainder via :meth:`drain` once that
    worker has joined. Single-consumer by contract — only the results worker calls
    :meth:`add`/:meth:`checkpoint`, and :meth:`drain` runs after it has exited — so the
    lockless extend/flush/clear is safe as long as that invariant holds.
    """

    def __init__(
        self,
        flush: Callable[[list[Subsidiary]], None],
        logger: logging.Logger,
        total: int,
    ) -> None:
        """Initialize the sink.

        Args:
            flush: Callable that persists a list of subsidiaries (e.g.
                :meth:`SubsidiaryPipeline.save_output`).
            logger: Logger for checkpoint progress/failure lines.
            total: Total documents to extract, for the checkpoint log line.
        """
        self._buffer: list[Subsidiary] = []
        self._flush = flush
        self._logger = logger
        self._total = total

    def add(self, batch: list[Subsidiary]) -> None:
        """Accumulate one processed document's extracted subsidiaries."""
        self._buffer.extend(batch)

    def checkpoint(self, extracted: int) -> None:
        """Flush the buffer to the output, dropping it from memory on success.

        Keeps the buffer bounded (~checkpoint interval) instead of growing to millions
        of rows over a run. The flush merges with the existing output, so flushed rows
        are not lost. A failed write is logged and swallowed, and the buffer is kept so
        its rows retry on the next checkpoint or the final :meth:`drain`.

        Args:
            extracted: Documents processed so far, for the log line.
        """
        if not self._buffer:
            return
        try:
            self._logger.info(
                "Checkpointing %d subsidiaries at %d / %d documents",
                len(self._buffer),
                extracted,
                self._total,
            )
            self._flush(self._buffer)
            self._buffer = []
        except Exception:
            self._logger.exception(
                "Checkpoint write failed at %d documents; keeping buffer for retry", extracted
            )

    def drain(self) -> list[Subsidiary]:
        """Return the rows not yet flushed and detach them from the sink.

        Called by :meth:`process` after the results worker has joined, so there is no
        concurrent mutation. Returns the tail accumulated since the last checkpoint
        (or the full set when checkpointing is disabled) for the final save.
        """
        remaining, self._buffer = self._buffer, []
        return remaining


class SubsidiaryPipeline(Pipeline):
    """Pipeline that fetches Exhibit 21 filings from SEC EDGAR and extracts subsidiary data."""

    CIK_JSON_URL = "https://data.sec.gov/submissions"
    _INPUT_SAMPLE_SIZE = int(os.environ.get("INPUT_SAMPLE_SIZE", "0"))
    _LOG_EVERY = 5
    # Coarser cadence for the input scan: it sweeps every candidate filing
    # (~100k for a full historical range), far more than the documents extracted.
    _LOAD_LOG_EVERY = 1000
    # Enqueued after all extraction tasks to tell the single results worker to
    # finish draining and exit.
    _RESULTS_SENTINEL = object()

    def __init__(
        self, config: PipelineConfig, sec_client: SecClient, extractor: GptExtractor
    ) -> None:
        """Initialize the subsidiary pipeline with failure registry.

        Args:
            config: Pipeline configuration including input/output paths, rate limit,
                worker count, and failure flush threshold.
            sec_client: Configured SEC EDGAR API client.
            extractor: Extractor used to parse subsidiary data from exhibit documents.
        """
        super().__init__(config, sec_client, extractor)
        self.failure_registry = FailureRegistry(
            config.failure_file,
            classifier=CorporateStructureFailureClassifier(),
            flush_every=config.failure_flush_every,
        )
        # Total exhibit documents to extract, known once load_input has selected
        # exhibits for every filing. Used as the fixed denominator in progress logs.
        self._total_documents = 0
        self._company_meta_cache: dict[str, CompanyMeta] = {}

    def _load_processed_accessions(self) -> set[str]:
        """Return accession numbers already present in the output parquet file.

        Returns:
            Set of accession numbers, or an empty set if the output file
            does not exist yet.
        """
        try:
            output_df = pd.read_parquet(self.config.output_file, columns=["accession_number"])
        except FileNotFoundError:
            return set()
        return set(output_df["accession_number"].unique())

    def _fetch_company_meta(self, cik: str) -> CompanyMeta:
        """Fetch per-CIK company metadata from the SEC submissions JSON.

        Cached per pipeline instance since ``load_input`` calls this once per
        filing — companies with multiple filings in the date range would
        otherwise trigger a redundant SEC request per extra filing. Safe
        without a lock: ``load_input`` runs single-threaded, before the
        extraction worker threads are started.

        Args:
            cik: SEC CIK number for the filer.

        Returns:
            CompanyMeta populated from the submissions endpoint, with blank
            defaults for any fields missing from a successful response.

        Raises:
            CompanyMetaFetchError: If the SEC request failed (the response
                contains an ``error`` key). A successful-but-sparse response is
                not an error and yields blank defaults.
        """
        if cik in self._company_meta_cache:
            return self._company_meta_cache[cik]

        cik_10 = str(int(cik)).zfill(10)
        url = f"{self.CIK_JSON_URL}/CIK{cik_10}.json"
        response = self.sec_client.query_endpoint(sec_url=url)
        if "error" in response:
            # Request failed after the SEC client's retries. Raise instead of caching
            # a blank CompanyMeta so the filing is skipped and retried on a later run.
            raise CompanyMetaFetchError(cik, url)
        data = response.get("data", {})
        biz = data.get("addresses", {}).get("business", {})
        company_meta = CompanyMeta(
            state_of_incorporation=data.get("stateOfIncorporation", ""),
            business_street1=biz.get("street1", ""),
            business_street2=biz.get("street2", ""),
            business_city=biz.get("city", ""),
            business_state=biz.get("stateOrCountry", ""),
            business_zip=biz.get("zipCode", ""),
            business_country=biz.get("country", ""),
            business_country_code=biz.get("countryCode", ""),
            tickers=tuple(t or "" for t in data.get("tickers") or ()),
            exchanges=tuple(e or "" for e in data.get("exchanges") or ()),
        )
        self._company_meta_cache[cik] = company_meta
        return company_meta

    @staticmethod
    def _select_exhibit_documents(
        scraped_filing: ScrapedFiling, exhibit_type: str
    ) -> tuple[ScrapedDocument, ...]:
        """Return the scraped documents matching the filing's exhibit type.

        Args:
            scraped_filing: Manifest whose documents are filtered.
            exhibit_type: Exhibit number to match — ``"21"`` or ``"8"``.

        Returns:
            Documents whose ``type`` starts with ``ex21``/``ex8`` (after
            stripping non-alphanumeric characters), in manifest order.
        """
        token = f"ex{exhibit_type}"  # ex21 or ex8
        return tuple(
            d
            for d in scraped_filing.documents
            if re.sub(r"[^0-9a-z]", "", d.type.lower()).startswith(token)
        )

    def _should_skip(self, filing: Filing, processed_accessions: set[str]) -> bool:
        """Return True if the filing was already processed or previously failed.

        Args:
            filing: Filing being considered for processing.
            processed_accessions: Accession numbers already present in the
                output file.

        Returns:
            True if the filing's accession number is in
            ``processed_accessions`` or already recorded in the failure
            registry.
        """
        return (
            filing.accession_number in processed_accessions
            or (filing.cik, filing.accession_number) in self.failure_registry
        )

    def load_input(self) -> list[Filing]:
        """Load input data from the SEC and return a list of filings.

        Filings with no matching exhibit documents are recorded as
        ``NO_EXHIBIT_FOUND`` failures and excluded from the returned list, so
        the count reflects filings that actually have exhibit content to fetch.

        Returns:
            A list of Filing objects
        """
        processed_accessions = self._load_processed_accessions()

        scraped_filings = iter_filings_by_form_type(
            form_types=TARGET_FORM_TYPES,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            bucket=self.config.sec_bucket,
            search_by="scraped_date",
        )

        if self._INPUT_SAMPLE_SIZE:
            scraped_filings = islice(scraped_filings, self._INPUT_SAMPLE_SIZE)

        filings = []
        for scraped_filing in scraped_filings:
            scanned = self.stats.increment("total_filing")
            if scanned % self._LOAD_LOG_EVERY == 0:
                self.logger.info(
                    "Scanned %d filings (%d with exhibits so far)", scanned, len(filings)
                )

            try:
                company_meta = self._fetch_company_meta(scraped_filing.cik)
            except CompanyMetaFetchError:
                # Retryable failure: not persisted to the registry, so the filing is
                # neither written to output nor skipped on the next run.
                self._record_failure(
                    (scraped_filing.cik, scraped_filing.accession_number),
                    FailureType.API_ERROR,
                    "warning",
                    "Failed to fetch company metadata for CIK %s - %s - will retry next run",
                    scraped_filing.cik,
                    scraped_filing.accession_number,
                    stat_keys=("failed_filings",),
                )
                continue

            filing = Filing(
                cik=scraped_filing.cik,
                filing_date=scraped_filing.filing_date,
                form_type=scraped_filing.form_type,
                accession_number=scraped_filing.accession_number,
                primary_document=scraped_filing.index_url,
                company_name=scraped_filing.company_name,
                company=company_meta,
            )
            filing.exhibit_documents = self._select_exhibit_documents(
                scraped_filing, filing.exhibit_type
            )

            if self._should_skip(filing, processed_accessions):
                self.stats.increment("skipped_filings")
                continue

            if not filing.exhibit_documents:
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.NO_EXHIBIT_FOUND,
                    "debug",
                    "No exhibit found for filing: %s - %s - %s (%s)",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    scraped_filing.index_url,
                    stat_keys=("failed_filings",),
                )
                continue

            filings.append(filing)

        self._total_documents = sum(len(f.exhibit_documents) for f in filings)
        self.logger.info(
            "Scan complete: %d filings scanned, %d with exhibits, %d documents to extract",
            self.stats.total_filing,
            len(filings),
            self._total_documents,
        )

        return filings

    def _record_failure(
        self,
        key: tuple[str, str],
        failure_type: FailureType,
        log_level: str,
        message: str,
        *log_args: object,
        stat_keys: tuple[str, ...] = ("failed_subsidiaries",),
    ) -> None:
        """Log a failure, increment stats, and register it in the failure registry.

        Args:
            key: Registry key tuple, typically ``(cik, filename)``.
            failure_type: Classified failure type.
            log_level: Logger method name (``"warning"``, ``"error"``, or
                ``"exception"``). ``"exception"`` behaves like ``"error"`` but
                also attaches the current traceback — only valid when called
                from within an active ``except`` block.
            message: ``%s``-style log message.
            *log_args: Arguments to substitute into ``message``.
            stat_keys: Stat field names to increment (default: ``("failed_subsidiaries",)``).
        """
        getattr(self.logger, log_level)(message, *log_args)
        for key_ in stat_keys:
            self.stats.increment(key_)
        self.failure_registry.add(key, failure_type)

    def _report_extraction(
        self,
        num_chunks: int,
        ungrounded_name: int,
        ungrounded_location: int,
        num_subsidiaries: int,
        filing: Filing,
    ) -> None:
        """Track stats on extraction operations.

        Args:
            num_chunks: The number of chunks and exhibit may be split up in
            ungrounded_name: The number of instances where name check failed
            ungrounded_location: The number of instances where location check failed
            num_subsidiaries: The number of subsidiaries extracted
            filing: The Filing object the subsidiaries were extracted for
        """
        self.stats.increment("total_subsidiaries", num_subsidiaries)

        if num_chunks > 1:
            self.stats.increment("chunked_extractions")

        if ungrounded_name:
            self.stats.increment("ungrounded_name", ungrounded_name)

        if ungrounded_location:
            self.stats.increment("ungrounded_location", ungrounded_location)

        if num_subsidiaries == 0:
            self._record_failure(
                (filing.cik, filing.accession_number),
                FailureType.NO_SUBSIDIARIES,
                "warning",
                "No subsidiaries found for filing: %s - %s - %s",
                filing.cik,
                filing.accession_number,
                filing.filing_date,
                stat_keys=("zero_subsidiaries",),
            )

    def _extract_worker(self, work_queue: queue.Queue, results_queue: queue.Queue) -> None:
        """Producer thread that extracts subsidiaries from queued exhibit documents.

        Runs as a daemon thread, consuming ``(filing, exhibit_contents)`` tuples from
        ``work_queue`` and posting each document's extracted ``list[Subsidiary]``
        batch to ``results_queue`` for the single results worker to accumulate and
        persist. Extraction errors are caught, logged, and recorded in the failure
        registry so the loop continues.

        One item is posted per processed document — an empty list on failure or a
        zero-subsidiary document — so the results worker can count documents (not
        just subsidiaries) and keep the ``extracted / total`` progress accurate.

        Args:
            work_queue: Queue of ``(Filing, dict)`` tuples to process. Each dict has
                ``"url"`` and ``"data"`` keys for the exhibit content.
            results_queue: Queue each document's ``list[Subsidiary]`` batch is posted
                to for the results worker to consume.

        Returns:
            None
        """
        while True:
            filing, exhibit_contents = work_queue.get()
            batch: list[Subsidiary] = []
            try:
                subsidiaries_batch, ungrounded_name, ungrounded_location, num_chunks = (
                    self.extractor.extract(filing, exhibit_contents)
                )
                self._report_extraction(
                    num_chunks=num_chunks,
                    ungrounded_name=ungrounded_name,
                    ungrounded_location=ungrounded_location,
                    num_subsidiaries=len(subsidiaries_batch),
                    filing=filing,
                )
                batch = subsidiaries_batch

            except DocumentError as e:
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.DOCUMENT_ERROR,
                    "error",
                    "Document error for filing: %s - %s - %s: %s @ %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    e,
                    exhibit_contents["url"],
                )

            except ExtractionTimeoutError:
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.TIMEOUT_ERROR,
                    "error",
                    "Timeout extracting subsidiaries from filing: %s - %s - %s @ %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    exhibit_contents["url"],
                    stat_keys=("failed_subsidiaries", "timeout_subsidiaries"),
                )

            except ExtractionTruncatedError as e:
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.TRUNCATED_ERROR,
                    "error",
                    "Truncated extraction for filing: %s - %s - %s: %s @ %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    e,
                    exhibit_contents["url"],
                    stat_keys=("failed_subsidiaries", "truncated_extractions"),
                )

            # Catch-all so one bad filing can't kill the worker; recorded as a failure.
            except Exception as e:  # noqa: BLE001
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.EXTRACTION_FAILED,
                    "exception",
                    "Error extracting subsidiaries from filing: %s - %s - %s @ %s: %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    exhibit_contents["url"],
                    e,
                )

            finally:
                # Post the batch (empty on failure) before marking the task done, so
                # work_queue.join() implies every batch is already on results_queue.
                results_queue.put(batch)
                work_queue.task_done()

    def _extract_pdf_text(self, raw_content: bytes, doc_url: str, filing: Filing) -> str:
        """Extract plain text from a PDF exhibit using pdfplumber.

        Args:
            raw_content: Raw PDF bytes fetched from S3.
            doc_url: Original SEC URL of the PDF, used for failure logging.
            filing: Filing the PDF belongs to, used for the failure registry key.

        Returns:
            Extracted text, or an empty string if the PDF could not be parsed.
        """
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(raw_content)) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:  # noqa: BLE001
            self._record_failure(
                (filing.cik, filing.accession_number),
                FailureType.NO_EXHIBIT_CONTENT,
                "error",
                "Failed to extract PDF content: %s",
                doc_url,
            )
        return text

    def _fetch_exhibit(self, filing: Filing) -> list[dict]:
        """Fetch exhibit data from the SEC.

        Args:
            filing: Filing object to fetch exhibit data from

        Returns:
            List of dicts with 'url' and 'data' keys
        """
        exhibit_content = []
        for doc in filing.exhibit_documents:
            if not doc.s3_key:
                continue

            try:
                raw_exhibit = load_content(doc.s3_key)
            except Exception as e:  # noqa: BLE001
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.NO_EXHIBIT_CONTENT,
                    "error",
                    "Failed to fetch exhibit %s - %s - %s from S3 (%s): %s",
                    doc.filename,
                    filing.cik,
                    filing.accession_number,
                    doc.s3_key,
                    e,
                )
                continue

            if not raw_exhibit:
                self._record_failure(
                    (filing.cik, filing.accession_number),
                    FailureType.NO_EXHIBIT_CONTENT,
                    "error",
                    "Exhibit %s - %s - %s does not have content (%s).",
                    doc.filename,
                    filing.cik,
                    filing.accession_number,
                    doc.s3_key,
                )
                continue

            ext = doc.filename.rsplit(".", 1)[-1].upper() if "." in doc.filename else ""
            if ext in ("HTM", "HTML", "TXT", "PDF"):
                self.stats.increment(f"{ext.lower()}_exhibits")

            if ext == "PDF":
                text = self._extract_pdf_text(raw_exhibit, doc.url, filing)
            elif ext in ("HTM", "HTML"):
                text = html_to_text(raw_exhibit.decode("utf-8", errors="replace"))
            else:
                text = raw_exhibit.decode("utf-8", errors="replace")

            if not text.strip():
                # Don't enqueue empty text - it would burn an OpenAI call downstream.
                # PDF parse failures are already recorded inside _extract_pdf_text;
                # record the other empty cases (e.g. HTML that renders to nothing).
                if ext != "PDF":
                    self._record_failure(
                        (filing.cik, filing.accession_number),
                        FailureType.NO_EXHIBIT_CONTENT,
                        "warning",
                        "Exhibit %s - %s - %s produced empty text (%s).",
                        doc.filename,
                        filing.cik,
                        filing.accession_number,
                        doc.s3_key,
                    )
                continue

            exhibit_content.append({"url": doc.url, "data": text})

        return exhibit_content

    def _results_worker(self, results_queue: queue.Queue, sink: "_ResultSink") -> None:
        """Single consumer that feeds result batches into ``sink`` and checkpoints it.

        Because only this thread drives ``sink``, the progress-log and checkpoint
        cadences run off one race-free counter with no locking. Consumes batches until
        it receives :attr:`_RESULTS_SENTINEL`, then returns; the final complete write is
        left to :meth:`save_output` in :meth:`run` (over ``sink.drain()``), so this
        worker only writes periodic checkpoints that make an interrupted run resumable
        (via :meth:`_load_processed_accessions` / :meth:`_should_skip`) without
        re-extracting — and re-paying for — completed documents.

        Args:
            results_queue: Queue of ``list[Subsidiary]`` batches (one per processed
                document), terminated by :attr:`_RESULTS_SENTINEL`.
            sink: Results sink this worker feeds; it owns the buffer and its flushes.

        Returns:
            None
        """
        while True:
            batch = results_queue.get()
            try:
                if batch is self._RESULTS_SENTINEL:
                    return
                sink.add(batch)
                extracted = self.stats.increment("extracted_documents")
                if extracted % self._LOG_EVERY == 0:
                    self.logger.info(
                        "Progress: %d extracted, %d queued, %d total documents",
                        extracted,
                        self.stats.queued_documents,
                        self._total_documents,
                    )
                if self.config.checkpoint_every and extracted % self.config.checkpoint_every == 0:
                    sink.checkpoint(extracted)
            finally:
                results_queue.task_done()

    def process(self, input_list: list[Filing]) -> list[Subsidiary]:
        """Fetch exhibit content and extract subsidiaries from each filing.

        Exhibit fetching runs on the main thread; extraction is parallelised across
        :attr:`~PipelineConfig.num_workers` daemon producer threads that post result
        batches to a queue drained by a single results worker. That consumer owns the
        results buffer and all output writes, so accumulation, progress logging (every
        :attr:`_LOG_EVERY` documents), and periodic checkpoints need no locking.
        Progress is logged periodically rather than via a live progress bar, since
        bars don't render correctly in aggregated cloud logs.

        Args:
            input_list: List of :class:`Filing` objects returned by
                :meth:`load_input`.

        Returns:
            List of :class:`Subsidiary` objects extracted across all filings.
        """
        work_queue = queue.Queue(maxsize=self.config.num_workers * 2)
        results_queue: queue.Queue = queue.Queue()
        sink = _ResultSink(flush=self.save_output, logger=self.logger, total=self._total_documents)

        # Producers extract in parallel; a single consumer owns the results buffer.
        extract_workers = [
            threading.Thread(
                target=self._extract_worker,
                args=(work_queue, results_queue),
                daemon=True,
                name=f"extract-worker-{i}",
            )
            for i in range(self.config.num_workers)
        ]
        for worker in extract_workers:
            worker.start()

        results_worker = threading.Thread(
            target=self._results_worker,
            args=(results_queue, sink),
            daemon=True,
            name="results-worker",
        )
        results_worker.start()

        # SEC operations to fetch exhibit data — one task per document
        for filing in input_list:
            exhibit_contents = self._fetch_exhibit(filing)
            for exhibit_content in exhibit_contents:
                work_queue.put((filing, exhibit_content))
                self.stats.increment("queued_documents")

        # Every producer posts its batch before task_done, so once work_queue drains
        # all batches are on results_queue. Signal end-of-stream and let the consumer
        # finish; run() performs the final, complete save_output over the drained tail.
        work_queue.join()
        results_queue.put(self._RESULTS_SENTINEL)
        results_worker.join()

        return sink.drain()

    def save_output(self, processed_list: list[Subsidiary]) -> None:
        """Deduplicate and persist extracted subsidiaries as a Parquet file.

        Merges new rows with any existing parquet, drops duplicates keyed on
        ``(parent_cik, accession_number, name)``, and stamps a UTC ``date_added``
        column before writing.

        Args:
            processed_list: List of :class:`Subsidiary` objects returned by
                :meth:`process`.

        Returns:
            None
        """
        if not processed_list:
            # pd.DataFrame([]) has zero columns (not just zero rows), so if no
            # output file exists yet, the location/parent_state_of_incorporation
            # normalization below would KeyError on a columnless frame. There's
            # nothing new to merge or normalize either way, so skip entirely.
            self.logger.info(
                "No subsidiaries to write; skipping save_output "
                "(all results already checkpointed, or none extracted)"
            )
            return

        # Save processed subsidiaries to a DataFrame
        subsidiaries_df = pd.DataFrame([dataclasses.asdict(s) for s in processed_list])

        try:
            existing_subsidiaries_df = pd.read_parquet(self.config.output_file)

            # Merge the existing subsidiaries with the new subsidiaries
            self.logger.info(
                "Merging existing %d subsidiaries with %d new subsidiaries",
                len(existing_subsidiaries_df),
                len(subsidiaries_df),
            )
            combined_subsidiaries_df = pd.concat(
                [existing_subsidiaries_df, subsidiaries_df], ignore_index=True
            )

        except FileNotFoundError:
            self.logger.info("No existing subsidiaries found, creating new file")
            combined_subsidiaries_df = subsidiaries_df

        # Canonicalize jurisdiction strings so the same place yields the same
        # value across filings. Applied to merged historic + new rows so that
        # alias-dict updates retroactively normalize older data on next write.
        combined_subsidiaries_df["location"] = (
            combined_subsidiaries_df["location"].fillna("").map(normalize_subsidiary_location)
        )
        combined_subsidiaries_df["parent_state_of_incorporation"] = (
            combined_subsidiaries_df["parent_state_of_incorporation"]
            .fillna("")
            .map(normalize_parent_location)
        )

        # Drop duplicate rows keyed on (parent_cik, accession_number, name)
        combined_subsidiaries_df = combined_subsidiaries_df.drop_duplicates(
            subset=["parent_cik", "accession_number", "name"]
        )

        # Add a date_added column if it doesn't exist and set the value to the current UTC timestamp
        if "date_added" not in combined_subsidiaries_df.columns:
            combined_subsidiaries_df["date_added"] = pd.NA
        combined_subsidiaries_df.loc[
            combined_subsidiaries_df["date_added"].isna(), "date_added"
        ] = datetime.datetime.now(datetime.UTC).isoformat()

        # Save the combined subsidiaries to the output file
        combined_subsidiaries_df.to_parquet(self.config.output_file)
        self.logger.info(
            "Saved %d subsidiaries to %s", len(combined_subsidiaries_df), self.config.output_file
        )

    def display_stats(self) -> None:
        """Log a formatted summary of pipeline statistics on completion.

        Writes filing totals (total, skipped, failed) and subsidiary totals
        (total, failed) to the logger at INFO level.

        Returns:
            None
        """
        self.logger.info("=" * 40)
        self.logger.info("Pipeline Stats")
        self.logger.info("=" * 40)
        self.logger.info("  Filings")
        self.logger.info("    Total:    %d", self.stats.total_filing)
        self.logger.info("    Skipped:  %d", self.stats.skipped_filings)
        self.logger.info("    Failed:   %d", self.stats.failed_filings)
        self.logger.info("  Subsidiaries")
        self.logger.info("    Total:    %d", self.stats.total_subsidiaries)
        self.logger.info("    Failed:   %d", self.stats.failed_subsidiaries)
        self.logger.info("    Timeouts: %d", self.stats.timeout_subsidiaries)
        self.logger.info("    Truncated: %d", self.stats.truncated_extractions)
        self.logger.info("    Chunked:   %d", self.stats.chunked_extractions)
        self.logger.info("    Zero:     %d", self.stats.zero_subsidiaries)
        self.logger.info("    Ungrounded name:     %d", self.stats.ungrounded_name)
        self.logger.info("    Ungrounded location: %d", self.stats.ungrounded_location)
        self.logger.info("    Dropped:             %d", self.stats.dropped_subsidiaries)
        self.logger.info("  Exhibits by type")
        self.logger.info("    HTM:  %d", self.stats.htm_exhibits)
        self.logger.info("    HTML: %d", self.stats.html_exhibits)
        self.logger.info("    TXT:  %d", self.stats.txt_exhibits)
        self.logger.info("    PDF:  %d", self.stats.pdf_exhibits)
        self.logger.info("=" * 40)

    def run(self) -> None:
        """Run the pipeline, flushing any buffered failures on completion."""
        try:
            super().run()
        finally:
            self.failure_registry.flush()
