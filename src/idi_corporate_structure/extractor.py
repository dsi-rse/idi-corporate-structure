"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
import html as _html
import importlib.resources
import json
import re
import statistics
from abc import ABC, abstractmethod

# Third-party imports
from bs4 import BeautifulSoup, Comment
from idi_ftm2j_shared.logs import get_logger

# Application imports
from idi_corporate_structure.api import OpenAiClient
from idi_corporate_structure.types import Filing, Subsidiary

_PROMPTS = importlib.resources.files("idi_corporate_structure.prompts")

_BLOCK_TAGS = frozenset(
    {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "table",
        "tr",
        "ul",
    }
)
_CELL_TAGS = frozenset({"td", "th"})
_ROW_STRUCTURE_TAGS = frozenset({"table", "tr"})
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")

_CHUNK_THRESHOLD_CHARS = 4_000
_CHUNK_MAX_CHARS = 4_000  # protects the model's input window
_CHUNK_MAX_ENTRIES = 75  # protects against per-chunk laziness
_CHUNK_OVERLAP_CHARS = 400

_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
# C0 control characters that str.split() does NOT treat as whitespace (so they
# survive into a name/document and break grounding). Excludes \t \n \r \x0b \x0c,
# which split() already collapses.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_PUNCT_RE = re.compile(r"[^\w]+")
_TRAILING_FOOTNOTE_RE = re.compile(r"\s*\(\d+\)\s*$")

# Grounding: a multi-word name whose tokens appear in order within this many
# characters is treated as present, recovering names that html_to_text shattered
# across table columns. Kept tight so scattered tokens don't pass the guard.
_SUBSEQUENCE_WINDOW_CHARS = 250
_MIN_SUBSEQUENCE_TOKENS = 2  # single-token names use strict/compact only


class DocumentError(Exception):
    """Exception raised for document-specific errors."""

    pass


class ExtractionTimeoutError(RuntimeError):
    """Exception raised when the OpenAI API times out during extraction."""

    pass


class ExtractionTruncatedError(RuntimeError):
    """Raised when the model's extraction response was cut off by the output token limit."""

    pass


class Extractor(ABC):
    """Interface for extracting subsidiaries from a single exhibit document."""

    @abstractmethod
    def extract(self, filing: Filing, document: dict) -> tuple[list[Subsidiary], int]:
        """Extract subsidiaries from a single exhibit document.

        Args:
            filing: The filing the document belongs to.
            document: Dict with 'url' and 'data' keys for the exhibit content.

        Returns:
            Tuple of (extracted Subsidiary objects, count of subsidiaries dropped
            during grounding checks).
        """
        ...


class GptExtractor(Extractor):
    """Extracts subsidiaries from a single exhibit document using GPT."""

    _DOCUMENT_ERROR_STATUS_CODE = 400
    _MAX_COMPLETION_TOKENS = 32768  # surfaces finish_reason="length" when truncation occurs
    _DEFAULT_MODEL = "gpt-4.1-nano"
    _LOW_YIELD_RATIO = 0.6  # per-chunk yield (output rows / input rows)
    # Re-extract a chunk whose yield is below this fraction of its sibling
    # chunks' median yield - a sharp outlier signals mid-document truncation,
    # whereas a uniformly low yield usually means input_rows is a poor
    # denominator for that layout (not real loss), so it is left alone.
    _SIBLING_OUTLIER_FACTOR = 0.6
    _RETRY_MAX_ENTRIES = 25  # smaller entry cap when re-extracting an outlier chunk
    _MIN_CHUNKS_FOR_RETRY = 2  # need at least one sibling to judge an outlier
    _SYSTEM_PROMPT: str = (
        _PROMPTS.joinpath("gpt_extractor_system.txt").read_text(encoding="utf-8").strip()
    )

    def __init__(self, openai_api_key: str, model: str = "") -> None:
        """Initialize the GPT extractor with the OpenAI API key.

        Args:
            openai_api_key: OpenAI API key.
            model: OpenAI model ID to use for extraction.  Defaults to
                ``_DEFAULT_MODEL`` when omitted or empty.
        """
        self._openai_client = OpenAiClient(api_key=openai_api_key)
        self._model = model or self._DEFAULT_MODEL
        self._logger = get_logger(type(self).__name__)

    def _extract_with_chunking(
        self, doc_text: str, company_name: str, doc_url: str
    ) -> tuple[list[dict], int]:
        """Run extraction one-shot, falling back to chunked extraction if needed.

        Two triggers cause chunking:
          * Preemptive: ``len(doc_text) > _CHUNK_THRESHOLD_CHARS`` (catches the
            ``finish_reason="stop"`` laziness case where the model gives up
            mid-extraction without surfacing as truncation).
          * Reactive: a one-shot call hits the explicit output cap and raises
            :class:`ExtractionTruncatedError`.

        Args:
            doc_text: Full plain-text exhibit content.
            company_name: String name of the filing company
            doc_url: SEC URL of the exhibit, included in chunk-yield log lines
                so a low-yield warning can be traced back to the source document.

        Returns:
            Tuple of (raw subsidiary dicts from the model, num chunks used).
            ``num_chunks == 1`` means no chunking was performed.
        """
        if len(doc_text) > _CHUNK_THRESHOLD_CHARS:
            return self._summarize_chunks(doc_text, company_name, doc_url)

        try:
            return self._summarize(doc_text).get("subsidiaries", []), 1
        except ExtractionTruncatedError:
            self._logger.info("One-shot extraction truncated; retrying with chunking")
            return self._summarize_chunks(doc_text, company_name, doc_url)

    @staticmethod
    def _chunk_input_rows(chunk: str) -> int:
        """Approximate the number of entity rows in a chunk (its paragraph count)."""
        return chunk.count("\n\n") + 1

    def _chunk_yield(self, chunk: str, chunk_subs: list[dict]) -> float:
        """Return output-rows / input-rows for a chunk (0.0 when it has no rows)."""
        input_rows = self._chunk_input_rows(chunk)
        return len(chunk_subs) / input_rows if input_rows else 0.0

    def _log_chunk(
        self,
        chunk: str,
        chunk_subs: list[dict],
        company_name: str,
        doc_url: str,
        i: int,
        num_chunks: int,
    ) -> None:
        """Log the chunking process.

        Args:
            chunk: The chunk of text to log.
            chunk_subs: The subsidiaries returned by the model for the chunk.
            company_name: The name of the filing company.
            doc_url: SEC URL of the exhibit the chunk was taken from.
            i: The index of the chunk.
            num_chunks: The number of chunks.
        """
        input_rows = self._chunk_input_rows(chunk)
        output_rows = len(chunk_subs)
        yield_ratio = self._chunk_yield(chunk, chunk_subs)
        log = self._logger.warning if yield_ratio < self._LOW_YIELD_RATIO else self._logger.info
        log(
            "%s chunk %d/%d: %d input rows → %d extracted (yield=%.2f) @ %s",
            company_name,
            i,
            num_chunks,
            input_rows,
            output_rows,
            yield_ratio,
            doc_url,
        )

    def _reextract_outlier(self, chunk: str) -> list[dict]:
        """Re-extract one low-yield chunk with a smaller entry cap.

        Splitting the chunk into fewer-entry sub-chunks counters the per-chunk
        "laziness" that makes the model return only part of a dense list. A
        sub-chunk that fails (truncation, timeout, or another API RuntimeError)
        is logged and skipped rather than raised — the retry is best-effort
        recovery layered on top of an already-successful original result, so it
        must never sink the overall extraction.

        Args:
            chunk: The chunk text to re-extract.

        Returns:
            Raw subsidiary dicts recovered from the sub-chunks (possibly empty).
        """
        sub_chunks = _chunk_document(
            chunk, _CHUNK_MAX_CHARS, _CHUNK_OVERLAP_CHARS, self._RETRY_MAX_ENTRIES
        )
        recovered: list[dict] = []
        for i, sub_chunk in enumerate(sub_chunks, start=1):
            try:
                recovered.extend(self._summarize(sub_chunk).get("subsidiaries", []))
            except RuntimeError as e:
                self._logger.error(
                    "Retry sub-chunk %d/%d (%d chars) failed (%s: %s); keeping partial recovery",
                    i,
                    len(sub_chunks),
                    len(sub_chunk),
                    type(e).__name__,
                    e,
                )
        return recovered

    def _retry_outlier_chunks(
        self,
        chunks: list[str],
        chunk_subs_list: list[list[dict]],
        company_name: str,
        doc_url: str,
    ) -> None:
        """Re-extract chunks whose yield is a sharp outlier vs their siblings.

        A per-chunk yield well below the sibling median signals mid-document
        truncation (the model returned far fewer rows than the chunk holds). A
        *uniformly* low yield, by contrast, usually means ``input_rows`` is a
        poor denominator for that layout rather than real loss, so it is not
        retried. Recovered rows are merged in and deduped by name.

        The final chunk is never retried: it typically carries trailing
        non-entity content (notes, legends, footnotes) that inflates its
        ``input_rows`` denominator, so a low yield there is usually an artifact
        rather than lost entities — re-extracting it just burns calls for no
        recovery.

        Mutates ``chunk_subs_list`` in place, replacing each outlier chunk's
        entry with its merged (original + recovered) subsidiaries. Returns
        nothing.

        Args:
            chunks: The chunk texts, in order.
            chunk_subs_list: Per-chunk extracted subsidiaries, aligned to
                ``chunks``; modified in place for outlier chunks.
            company_name: Filing company name, for log lines.
            doc_url: SEC URL of the exhibit, for log lines.
        """
        if len(chunks) < self._MIN_CHUNKS_FOR_RETRY:
            return

        yields = [self._chunk_yield(c, subs) for c, subs in zip(chunks, chunk_subs_list)]
        last_index = len(chunks) - 1

        for i, (chunk, subs) in enumerate(zip(chunks, chunk_subs_list)):
            sibling_median = statistics.median([y for j, y in enumerate(yields) if j != i])
            is_outlier = (
                i != last_index  # trailing notes make the final chunk a false outlier
                and yields[i] < self._LOW_YIELD_RATIO
                and sibling_median >= self._LOW_YIELD_RATIO
                and yields[i] < self._SIBLING_OUTLIER_FACTOR * sibling_median
            )
            if not is_outlier:
                continue

            self._logger.warning(
                "%s chunk %d/%d yield %.2f is a sharp outlier (sibling median %.2f) - "
                "re-extracting @ %s",
                company_name,
                i + 1,
                len(chunks),
                yields[i],
                sibling_median,
                doc_url,
            )
            merged = dedup_by_name(subs + self._reextract_outlier(chunk))
            self._logger.info(
                "%s chunk %d/%d retry: %d → %d rows after merge @ %s",
                company_name,
                i + 1,
                len(chunks),
                len(subs),
                len(merged),
                doc_url,
            )
            chunk_subs_list[i] = merged

    def _summarize_chunks(
        self, doc_text: str, company_name: str, doc_url: str
    ) -> tuple[list[dict], int]:
        """Chunk ``doc_text`` and run a separate summarize call per chunk.

        Args:
            doc_text: Full plain-text exhibit content
            company_name: String name of the filing company
            doc_url: SEC URL of the exhibit, included in log lines for traceability.

        Returns:
            Tuple of (concatenated raw subsidiaries from all chunks, chunk count)

        Raises:
            ExtractionTruncatedError: If any individual chunk truncates — the
                chunk-size constants need tuning down.
        """
        chunks = _chunk_document(
            doc_text, _CHUNK_MAX_CHARS, _CHUNK_OVERLAP_CHARS, _CHUNK_MAX_ENTRIES
        )
        self._logger.info(
            "%s chunked extraction: %d chunks @ %s", company_name, len(chunks), doc_url
        )

        chunk_subs_list: list[list[dict]] = []
        for i, chunk in enumerate(chunks, 1):
            try:
                result = self._summarize(chunk)
            except ExtractionTruncatedError:
                self._logger.error(
                    "Chunk %d/%d truncated — chunk size may be too large @ %s",
                    i,
                    len(chunks),
                    doc_url,
                )
                raise

            chunk_subs = result.get("subsidiaries", [])
            self._log_chunk(chunk, chunk_subs, company_name, doc_url, i, len(chunks))
            chunk_subs_list.append(chunk_subs)

        self._retry_outlier_chunks(chunks, chunk_subs_list, company_name, doc_url)

        all_subs = [sub for chunk_subs in chunk_subs_list for sub in chunk_subs]
        return all_subs, len(chunks)

    def _get_request_data_json(self, document: str) -> dict:
        """Build the OpenAI chat completions request payload for subsidiary extraction.

        Constructs a structured-output request that instructs the model to parse
        a subsidiary table from ``document`` and return a JSON object conforming
        to the ``list_of_subsidiaries`` schema.

        Args:
            document: Raw exhibit text (Markdown or plain text) to send as the user
                message.

        Returns:
            Dict ready to be serialized and posted to the OpenAI API, containing
            ``model``, ``messages``, and ``response_format`` keys.
        """
        return {
            "model": self._model,
            "max_completion_tokens": self._MAX_COMPLETION_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": self._SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": document,
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "list_of_subsidiaries",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subsidiaries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "location": {"type": ["string", "null"]},
                                        "source_quote": {"type": "string"},
                                    },
                                    "required": ["name", "location", "source_quote"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        }

    def _summarize(self, document: str) -> dict:
        """Send the document to GPT and return the parsed subsidiary data.

        Args:
            document: Raw exhibit text to pass to the model.

        Returns:
            Parsed JSON response from the model as a dict with a ``"subsidiaries"``
            key containing a list of ``{"name": ..., "location": ..., "source_quote": ...}`` objects.

        Raises:
            DocumentError: If the API returns a 400-level status code, indicating
                the document itself is malformed or too long.
            ExtractionTimeoutError: If the API call timed out.
            ExtractionTruncatedError: If the model response was cut off by the
                output token limit (``finish_reason == "length"``).
            RuntimeError: If the API returns any other error.
        """
        post_data = self._get_request_data_json(document)
        response = self._openai_client.query_endpoint(post_data)

        if "error" in response:
            if response.get("status_code") == self._DOCUMENT_ERROR_STATUS_CODE:
                raise DocumentError(response["error"])
            if response.get("timeout"):
                raise ExtractionTimeoutError(response["error"])
            raise RuntimeError(response["error"])

        choice = response["data"]["choices"][0]
        finish_reason = choice.get("finish_reason")
        usage = response["data"].get("usage", {})
        self._logger.debug(
            "OpenAI extraction input_chars=%d | finish_reason=%s | usage=%s",
            len(document),
            finish_reason,
            usage,
        )

        if finish_reason == "length":
            raise ExtractionTruncatedError(
                f"Model response truncated at output token limit "
                f"(max_completion_tokens={self._MAX_COMPLETION_TOKENS}, usage={usage})"
            )

        content = choice["message"]["content"]
        return json.loads(content)

    def _is_location_grounded(
        self, name: str, location: str, doc_text_normalized: str, doc_url: str
    ) -> int:
        """Check if a location is near a name in the document and logs a warning if not.

        Args:
            name: The name to check.
            location: The location to check.
            doc_text_normalized: The normalized document text to check for the name and location.
            doc_url: The URL of the document.

        Returns:
            The number of ungrounded locations.
        """
        ungrounded_location = 0
        if location:
            normalized_name = _normalize(name)
            name_pos = doc_text_normalized.find(normalized_name)
            if name_pos == -1:
                # Name only matched via the compact (punctuation-stripped) fallback
                # - using -1 as-is would silently check the start of the
                # document instead. Skip rather than report a bogus result.
                self._logger.debug(
                    "Skipping location-proximity check for %r (grounded via compact "
                    "fallback, no exact position) @ %s",
                    name,
                    doc_url,
                )
                return ungrounded_location

            name_len = len(normalized_name)
            window = doc_text_normalized[max(0, name_pos - 200) : name_pos + name_len + 200]

            if _normalize(location) not in window:
                self._logger.debug("Location %r not near name %r @ %s", location, name, doc_url)
                ungrounded_location += 1
        return ungrounded_location

    def _locate_grounded_subsidiaries(
        self, subsidiaries: list[dict], document: dict
    ) -> tuple[list[dict], int, int]:
        """Locate subsidiaries whose names are grounded in the document.

        The document is the source of truth. A subsidiary is kept if its
        ``name`` appears in the document. The model's ``source_quote`` is
        advisory: a mismatch between quote and document is logged at DEBUG
        level but does not drop the row.

        Args:
            subsidiaries: The subsidiaries returned by the model.
            document: Dict with ``"url"`` and ``"data"`` (text) keys.

        Returns:
            Tuple of (kept subsidiaries, count dropped for missing name, count dropped for missing location).
        """
        grounded_subsidiaries = []
        ungrounded_name = 0
        ungrounded_location = 0

        doc_text = document.get("data", "")
        doc_url = document.get("url", "")
        # Normalize the document once and reuse across every name, rather than
        # re-normalizing the whole exhibit per subsidiary.
        doc_text_normalized = _normalize(doc_text)
        doc_text_compact = _compact(doc_text)

        for sub in subsidiaries:
            name = sub.get("name", "")
            if not _is_name_grounded(name, doc_text_normalized, doc_text_compact):
                self._logger.warning("Dropped %r from %s (name not in document)", name, doc_url)
                ungrounded_name += 1
                continue

            quote = sub.get("source_quote", "")
            if quote and _normalize(quote) not in doc_text_normalized:
                self._logger.debug("Quote not in document for %r @ %s", name, doc_url)

            ungrounded_location += self._is_location_grounded(
                name, sub.get("location") or "", doc_text_normalized, doc_url
            )
            grounded_subsidiaries.append(sub)

        if ungrounded_name:
            self._logger.warning(
                "Dropped %d ungrounded subsidiaries from %s", ungrounded_name, doc_url
            )

        return grounded_subsidiaries, ungrounded_name, ungrounded_location

    def extract(self, filing: Filing, document: dict) -> tuple[list[Subsidiary], int, int, int]:
        """Extract subsidiaries from an exhibit document using GPT.

        Sends the document text to the OpenAI API (one-shot or chunked) and maps
        each returned item to a :class:`Subsidiary` dataclass, inheriting parent
        metadata from ``filing``.

        Args:
            filing: The SEC filing the exhibit belongs to. Provides parent company
                metadata (CIK, name, location, dates).
            document: Dict with ``"url"`` (exhibit URL) and ``"data"`` (exhibit text)
                keys.

        Returns:
            Tuple of (extracted Subsidiary objects, ungrounded-name count,
            ungrounded-location count, num chunks used). ``num_chunks > 1``
            indicates chunked extraction was used.

        Raises:
            DocumentError: If GPT rejects the document (e.g. content too long or
                malformed).
            ExtractionTruncatedError: If a chunked extraction still truncates
                (chunk size needs tuning).
            RuntimeError: If the OpenAI API returns any other error.
        """
        # Summarize the subsidiaries in the exhibit
        raw_subs, num_chunks = self._extract_with_chunking(
            document["data"], filing.company_name, document["url"]
        )

        # Dedupe by normalized name
        deduped = dedup_by_name(raw_subs=raw_subs)

        # Double check the results of the summarize
        grounded_subsidiaries, ungrounded_name, ungrounded_location = (
            self._locate_grounded_subsidiaries(deduped, document)
        )

        # Create subsidiaries
        subsidiaries = [
            Subsidiary(
                parent_cik=filing.cik,
                parent_name=filing.company_name,
                parent_state_of_incorporation=filing.company.state_of_incorporation,
                parent_business_street1=filing.company.business_street1,
                parent_business_street2=filing.company.business_street2,
                parent_business_city=filing.company.business_city,
                parent_business_state=filing.company.business_state,
                parent_business_zip=filing.company.business_zip,
                parent_business_country=filing.company.business_country,
                parent_business_country_code=filing.company.business_country_code,
                parent_tickers=",".join(filing.company.tickers),
                parent_exchanges=",".join(filing.company.exchanges),
                filing_date=filing.filing_date,
                period_of_report=filing.period_of_report,
                form_type=filing.form_type,
                exhibit_type=filing.exhibit_type,
                accession_number=filing.accession_number,
                exhibit_url=document["url"],
                name=_clean_name(sub["name"]),
                location=sub.get("location") or "",
                source_quote=sub.get("source_quote") or "",
            )
            for sub in grounded_subsidiaries
        ]
        return subsidiaries, ungrounded_name, ungrounded_location, num_chunks


def html_to_text(raw_html: str) -> str:
    """Convert HTML to plain text, preserving table row/cell boundaries.

    Args:
        raw_html: Raw HTML string.

    Returns:
        Plain text with block tags rendered as newlines and table cells
        separated by spaces. HTML entities are decoded. If the input is
        already plain text (no tags), it passes through with whitespace
        normalized.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        if tag.name in _CELL_TAGS:
            tag.insert_before(" ")
            tag.insert_after(" ")
        elif tag.name in _BLOCK_TAGS:
            # Filing software commonly wraps each cell's text in its own block
            # tag purely for styling (e.g. <td><div>...</div></td>).
            # Only "table"/"tr" force a break even when nested in a cell, so a
            # genuinely nested table still gets one.
            if tag.name not in _ROW_STRUCTURE_TAGS and tag.find_parent(_CELL_TAGS) is not None:
                tag.insert_before(" ")
                tag.insert_after(" ")
            else:
                tag.insert_before("\n")
                tag.insert_after("\n")

    text = _html.unescape(soup.get_text())

    lines = [_INLINE_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    collapsed = _MULTINEWLINE_RE.sub("\n\n", "\n".join(lines))
    return collapsed.strip()


def _take_overlap(chunk_text: str, overlap_chars: int) -> str:
    r"""Return the trailing whole paragraphs of ``chunk_text`` fitting in ``overlap_chars``.

    Used to build the overlap region between adjacent chunks. By taking only
    whole paragraphs, we guarantee the overlap text never bisects an entity row
    — so the next chunk never begins with a partial name like ``"wer Eight
    Project LLC"`` that the model would mis-extract as ``"Eight Project LLC"``.

    Always returns at least the final paragraph, even if it exceeds
    ``overlap_chars``, so the boundary entity is always carried forward.

    Args:
        chunk_text: The just-emitted chunk's full text.
        overlap_chars: Soft budget for the overlap region.

    Returns:
        Joined paragraphs (with ``\n\n`` separators) or empty string when
        ``overlap_chars <= 0``.
    """
    if overlap_chars <= 0:
        return ""

    paragraphs = chunk_text.split("\n\n")
    overlap_paras: list[str] = []
    overlap_len = 0

    for para in reversed(paragraphs):
        # +2 accounts for the "\n\n" separator we'd add when joining
        if overlap_paras and overlap_len + len(para) + 2 > overlap_chars:
            break
        overlap_paras.insert(0, para)
        overlap_len += len(para) + 2

    return "\n\n".join(overlap_paras)


def _chunk_document(
    text: str,
    max_chars: int,
    overlap_chars: int,
    max_entries: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks capped by character count and entry count.

    A chunk closes when the next paragraph would exceed ``max_chars`` OR the
    chunk already holds ``max_entries`` paragraphs. The entry cap prevents
    dense tables from packing too many rows into one chunk, which causes the
    model to silently drop entries. Each chunk carries ``overlap_chars`` of
    trailing text from the previous chunk; duplicates are removed by name.

    Args:
        text: Plain-text exhibit content to chunk.
        max_chars: Soft maximum characters per chunk.
        overlap_chars: Trailing characters carried into the next chunk.
        max_entries: Soft maximum paragraphs per chunk. ``None`` disables it.

    Returns:
        Ordered list of chunk strings. Returns ``[text]`` if no split needed.
    """
    if len(text) <= max_chars and (max_entries is None or text.count("\n\n") + 1 <= max_entries):
        return [text]

    logger = get_logger(__name__)
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for "\n\n" separator

        # Handle oversized paragraphs
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            logger.warning(
                "Paragraph exceeds chunk size (%d > %d chars); emitting as single oversized chunk",
                len(para),
                max_chars,
            )
            chunks.append(para)
            continue

        # Handle character and entry caps
        would_exceed_chars = current_len + para_len > max_chars
        would_exceed_entries = max_entries is not None and len(current) >= max_entries
        if (would_exceed_chars or would_exceed_entries) and current:
            chunks.append("\n\n".join(current))
            tail = _take_overlap(
                chunks[-1], overlap_chars
            )  # carry over overlap from previous chunk
            current = [tail] if tail else []  # start new chunk with overlap
            current_len = len(tail)

        current.append(para)  # add current paragraph to current chunk
        current_len += para_len

    if current:  # add last chunk if not empty
        chunks.append("\n\n".join(current))

    return chunks


def _compact(s: str) -> str:
    """Aggressive normalization: lowercase + strip all non-alphanumerics.

    Used as a fallback grounding check for names the model returned with
    minor punctuation or whitespace differences from the exhibit.

    Args:
        s: String to compact.

    Returns:
        Lowercased string with all non-word characters removed.
    """
    return _PUNCT_RE.sub("", _normalize(s))


def _name_tokens_in_window(name: str, document_normalized: str) -> bool:
    """Return True if the name's tokens appear in order within a bounded window.

    Recovers multi-word names that ``html_to_text`` shattered across table
    columns: when a wrapped name cell is linearized row-major, the row's other
    columns land between the name's words, so the full name is not a contiguous
    substring even though every word is present, in order, and close together.

    Tokens are matched in order, and each must *start* within
    ``_SUBSEQUENCE_WINDOW_CHARS`` of where the first token starts (the bound is
    on start offsets, so a token may extend past the window as long as it begins
    inside it). A fabricated name whose words are scattered far apart is
    therefore still rejected — this preserves the hallucination guard.
    Single-token names are not eligible: strict/compact already cover a lone
    token, and admitting it here would only loosen the check.

    Args:
        name: Candidate subsidiary name.
        document_normalized: The already-``_normalize``d document text.

    Returns:
        True if the name's tokens form an in-order, windowed subsequence.
    """
    tokens = _normalize(name).split()
    if len(tokens) < _MIN_SUBSEQUENCE_TOKENS:
        return False

    start = 0
    while True:
        first = document_normalized.find(tokens[0], start)
        if first == -1:
            return False
        pos = first + len(tokens[0])
        matched = True
        for token in tokens[1:]:
            nxt = document_normalized.find(token, pos)
            if nxt == -1:
                # Token absent from the rest of the doc; sliding the first token
                # forward only pushes the search later, so no match can exist.
                return False
            if nxt - first > _SUBSEQUENCE_WINDOW_CHARS:
                # Found but out of window; a later occurrence of the first token
                # may pull the span back within the window, so retry.
                matched = False
                break
            pos = nxt + len(token)
        if matched:
            return True
        start = first + 1


def _is_name_grounded(name: str, document_normalized: str, document_compact: str) -> bool:
    """Check if a name is grounded against pre-normalized document forms.

    Tries a strict normalized substring match first. If that fails, falls
    back to a compact (punctuation-stripped) match to catch cases where the
    model returned a name with minor punctuation or whitespace differences
    from the exhibit (e.g. dropped parentheses, "Health Care" vs
    "Healthcare"). Finally, for multi-word names, falls back to a windowed
    in-order token match to recover names shattered across table columns by
    ``html_to_text`` linearization.

    The document forms are passed in so the caller can compute them once per
    exhibit and reuse them across every candidate name, rather than
    re-normalizing the whole document for each subsidiary.

    Args:
        name: The name to check.
        document_normalized: The document text after ``_normalize``.
        document_compact: The document text after ``_compact``.

    Returns:
        True if the name is found via the strict, compact, or windowed match,
        False otherwise.
    """
    if not name:
        return False

    if _normalize(name) in document_normalized:
        return True

    if _compact(name) in document_compact:
        get_logger(__name__).debug(
            "Name %r matched via compact fallback (model produced a slight variant)", name
        )
        return True

    if _name_tokens_in_window(name, document_normalized):
        get_logger(__name__).debug(
            "Name %r matched via windowed-subsequence fallback (table-shattered layout)", name
        )
        return True

    return False


def _is_name_in_document(name: str, document: str) -> bool:
    """Check if a name appears in the document, normalizing the document inline.

    Thin convenience wrapper over :func:`_is_name_grounded` for callers with a
    single name to check (and the test suite). Hot loops over many names should
    call :func:`_is_name_grounded` with document forms computed once.

    Args:
        name: The name to check.
        document: The raw document text to check for the name.

    Returns:
        True if the name is found via the strict, compact, or windowed match.
    """
    return _is_name_grounded(name, _normalize(document), _compact(document))


def _normalize(s: str) -> str:
    r"""Decode HTML entities, normalize apostrophes and whitespace, and lowercase.

    Zero-width and control characters are stripped so they cannot wedge
    themselves between a name's words and defeat the grounding substring match
    (e.g. an exhibit that renders ``VisitIQ,\r2LLC`` or ``USCNHK Group\x01...``).
    """
    s = _html.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = _INVISIBLE_CHARS_RE.sub("", s)
    s = _CONTROL_CHARS_RE.sub(" ", s)
    return " ".join(s.split()).lower()


def dedup_by_name(raw_subs: list[dict]) -> list[dict]:
    """Dedupe by normalized name (collisions come from the chunk overlap region)

    Args:
        raw_subs: List of dictionaries that contain extracted subsidiaries

    Returns:
        De-duplicated subsidiaries for exhibit
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for sub in raw_subs:
        key = _normalize(sub.get("name", ""))
        # First occurrence wins for location and source_quote
        if key and key not in seen:
            seen.add(key)
            deduped.append(sub)
    return deduped


def _clean_name(name: str) -> str:
    """Clean up subsidiary name.

    Names are kept verbatim through extraction and grounding (see the system
    prompt) so exhibit footnote markers like the trailing ``(1)`` in
    "Freedom Bank Kazakhstan JSC, Kazakhstan(1)" survive into the model's
    output. Grounding has already run by the time this is called, so it's
    safe to strip that footnote noise here — mirrors the trailing-footnote
    strip already applied to ``location`` in normalization.py.

    Args:
        name: String extracted for subsidiary name

    Returns:
        cleaned name string
    """
    name = _html.unescape(name)
    name = _INVISIBLE_CHARS_RE.sub("", name)
    name = name.replace("\xa0", " ")
    name = " ".join(name.split())
    return _TRAILING_FOOTNOTE_RE.sub("", name).strip()
