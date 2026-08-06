"""Tests for processor.extractor — GptExtractor."""

import json
from pathlib import Path

import pytest

from idi_corporate_structure.extractor import (
    ExtractionTimeoutError,
    ExtractionTruncatedError,
    GptExtractor,
    html_to_text,
)
from idi_corporate_structure.types import Subsidiary
from tests.conftest import make_exhibit_response

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_JNJ_FIXTURE = _FIXTURES_DIR / "jnj_ex21_subsidiaries.htm"

# Convenience quote constants matching make_exhibit_response() default content.
_QUOTE_APPLE_OPS = "Apple Operations LLC (Delaware)"
_QUOTE_APPLE_EU = "Apple Europe Ltd (Ireland)"


def _make_openai_response(subsidiaries: list[dict], finish_reason: str = "stop") -> dict:
    """Build a fake OpenAI chat completions response dict."""
    return {
        "status_code": 200,
        "url": "https://api.openai.com/v1/chat/completions",
        "data": {
            "choices": [
                {
                    "message": {"content": json.dumps({"subsidiaries": subsidiaries})},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
    }


class TestGptExtractor:
    """Tests for GptExtractor.extract()."""

    def test_returns_subsidiaries_from_gpt_response(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "Apple Europe Ltd",
                        "location": "Ireland",
                        "source_quote": _QUOTE_APPLE_EU,
                    },
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 2
        assert all(isinstance(s, Subsidiary) for s in result)
        assert result[0].name == "Apple Operations LLC"
        assert result[0].location == "Delaware"
        assert result[0].source_quote == _QUOTE_APPLE_OPS
        assert result[1].name == "Apple Europe Ltd"
        assert result[1].location == "Ireland"
        assert result[1].source_quote == _QUOTE_APPLE_EU

    def test_subsidiary_filing_fields_preserved(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())
        s = result[0]

        assert s.parent_cik == sample_filing.cik
        assert s.parent_name == sample_filing.company_name
        assert s.filing_date == sample_filing.filing_date
        assert s.period_of_report == sample_filing.period_of_report
        assert s.form_type == sample_filing.form_type
        assert s.accession_number == sample_filing.accession_number

    def test_exhibit_url_from_document(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        document = make_exhibit_response()
        result, *_ = extractor.extract(sample_filing, document)

        assert result[0].exhibit_url == document["url"]

    def test_null_location_becomes_empty_string(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": None,
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].location == ""
        assert result[0].source_quote == _QUOTE_APPLE_OPS

    def test_returns_empty_list_for_no_subsidiaries(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([]),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_raises_on_api_error(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value={"error": "connection refused"},
        )

        with pytest.raises(RuntimeError):
            extractor.extract(sample_filing, make_exhibit_response())

    def test_raises_extraction_timeout_error_on_timeout(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value={"error": "Timeout querying https://api.openai.com/...", "timeout": True},
        )

        with pytest.raises(ExtractionTimeoutError):
            extractor.extract(sample_filing, make_exhibit_response())

    # ── Truncation tests ──────────────────────────────────────────────────────

    def test_raises_truncated_error_on_finish_reason_length(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ],
                finish_reason="length",
            ),
        )

        with pytest.raises(ExtractionTruncatedError):
            extractor.extract(sample_filing, make_exhibit_response())

    def test_finish_reason_stop_parses_normally(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ],
                finish_reason="stop",
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_request_payload_sets_max_completion_tokens(self):
        extractor = GptExtractor(openai_api_key="fake-key")
        payload = extractor._get_request_data_json("some document text")

        assert "max_completion_tokens" in payload
        assert payload["max_completion_tokens"] == GptExtractor._MAX_COMPLETION_TOKENS

    # ── Grounding tests ───────────────────────────────────────────────────────

    def test_source_quote_preserved(self, sample_filing, mocker):
        """source_quote from GPT response is mapped onto the Subsidiary."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].source_quote == _QUOTE_APPLE_OPS

    def test_name_missing_from_doc_is_dropped(self, sample_filing, mocker):
        """Subsidiaries whose name does not appear in the document are dropped."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Made Up Corp",
                        "location": "Delaware",
                        "source_quote": "Made Up Corp (Delaware)",
                    }
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_grounded_rows_kept_ungrounded_dropped(self, sample_filing, mocker):
        """Rows whose name appears in the doc are kept; missing names are dropped."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "Ghost Corp",
                        "location": "Nevada",
                        "source_quote": "Ghost Corp (Nevada)",
                    },
                ]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_empty_source_quote_kept_when_name_in_doc(self, sample_filing, mocker):
        """An empty source_quote is not a drop reason when the name is grounded."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": "Apple Operations LLC", "location": "Delaware", "source_quote": ""}]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_html_entity_name_matches_decoded_doc(self, sample_filing, mocker):
        """HTML entities in the model's name still match a decoded plain-text doc."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "&#8220;American Eagle&#8221; Holdings",
                        "location": "Delaware",
                        "source_quote": "&#8220;American Eagle&#8221; Holdings",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content='"American Eagle" Holdings, a Delaware Corporation')
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1
        assert result[0].name == "\u201cAmerican Eagle\u201d Holdings"

    def test_smart_quote_name_matches_ascii_doc(self, sample_filing, mocker):
        """A smart-quoted name in the model output matches an ASCII doc."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "O\u2019Reilly Holdings",
                        "location": "Delaware",
                        "source_quote": "O\u2019Reilly Holdings",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content="O'Reilly Holdings, a Delaware Corporation")
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1

    def test_name_surrounding_whitespace_matches_doc(self, sample_filing, mocker):
        """Normalization tolerates runs of whitespace between tokens."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": "Apple Operations LLC",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content="Apple  Operations   LLC\n(Delaware)")
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1

    def test_quote_mismatch_does_not_drop_but_logs_debug(self, sample_filing, mocker):
        """A non-matching source_quote logs DEBUG but does not drop the row."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": "fabricated quote that is not in doc",
                    }
                ]
            ),
        )
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert any(
            "Quote not in document" in str(call.args[0]) for call in debug_spy.call_args_list
        )

    def test_location_not_near_name_logs_debug(self, sample_filing, mocker):
        """A location absent from the window around the name logs DEBUG but keeps the row."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Narnia",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert any(
            "Location" in str(call.args[0]) and "not near name" in str(call.args[0])
            for call in debug_spy.call_args_list
        )

    def test_location_near_name_does_not_log_debug(self, sample_filing, mocker):
        """A location present in the window around the name emits no location debug log."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert not any("Location" in str(call.args[0]) for call in debug_spy.call_args_list)

    def test_is_name_in_document_normalizes_whitespace(self):
        """_is_name_in_document matches despite whitespace differences."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Apple  Operations   LLC\n(Delaware)"
        assert _is_name_in_document("Apple Operations LLC", doc)

    def test_is_name_in_document_rejects_missing_name(self):
        """_is_name_in_document returns False when the name is absent."""
        from idi_corporate_structure.extractor import _is_name_in_document

        assert not _is_name_in_document("Ghost Corp", "Apple Operations LLC (Delaware)")


class TestChunkDocument:
    """Tests for the _chunk_document splitter."""

    def test_small_doc_returns_single_chunk(self):
        from idi_corporate_structure.extractor import _chunk_document

        text = "small doc"
        chunks = _chunk_document(text, max_chars=1000, overlap_chars=100)

        assert chunks == [text]

    def test_large_doc_returns_multiple_chunks(self):
        from idi_corporate_structure.extractor import _chunk_document

        paragraphs = [f"para{i} " + "x" * 100 for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=500, overlap_chars=50)

        assert len(chunks) > 1
        # Every chunk respects the size limit (allowing for overlap headroom)
        for c in chunks:
            assert len(c) <= 500 + 50

    def test_chunks_split_on_paragraph_boundaries(self):
        from idi_corporate_structure.extractor import _chunk_document

        paragraphs = [f"Paragraph {i}" for i in range(50)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=80, overlap_chars=0)

        # Every chunk should consist of complete paragraphs joined by \n\n
        for chunk in chunks:
            for line in chunk.split("\n\n"):
                stripped = line.strip()
                # Each line is either a "Paragraph N" entry or empty
                assert stripped == "" or stripped.startswith("Paragraph ")

    def test_chunks_overlap_with_whole_paragraphs(self):
        """Overlap is paragraph-aligned.

        Each non-first chunk starts with one or more complete paragraphs that
        also appeared at the end of the previous chunk.
        """
        from idi_corporate_structure.extractor import _chunk_document

        paragraphs = [f"para{i}_" + "x" * 100 for i in range(15)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=400, overlap_chars=150)

        assert len(chunks) > 1
        for i in range(1, len(chunks)):
            prev_paras = chunks[i - 1].split("\n\n")
            curr_paras = chunks[i].split("\n\n")
            # The first paragraph of chunk i must equal the last paragraph of chunk i-1.
            # That guarantees overlap text never bisects an entity row.
            assert curr_paras[0] == prev_paras[-1], (
                f"Chunk {i} should start with the last paragraph of chunk {i - 1}, "
                f"but got {curr_paras[0]!r} vs {prev_paras[-1]!r}"
            )

    def test_oversized_paragraph_emitted_as_own_chunk(self):
        from idi_corporate_structure.extractor import _chunk_document

        text = "small\n\n" + "x" * 5000 + "\n\nsmall"
        chunks = _chunk_document(text, max_chars=1000, overlap_chars=100)

        # The oversized middle paragraph must appear as its own chunk
        assert any(len(c) > 1000 for c in chunks)

    def test_max_entries_cap_splits_dense_chunk(self):
        """A char-budget-friendly run of many short paragraphs splits on the entry cap.

        Models get lazy and silently drop entries when a single chunk asks
        them to enumerate >75 short rows, even if the chunk easily fits in the
        char window. This test pins the entry cap as a separate dimension.
        """
        from idi_corporate_structure.extractor import _chunk_document

        paragraphs = [f"Para{i}" for i in range(200)]
        text = "\n\n".join(paragraphs)

        no_cap = _chunk_document(text, max_chars=10_000, overlap_chars=0)
        capped = _chunk_document(text, max_chars=10_000, overlap_chars=0, max_entries=75)

        assert len(no_cap) == 1, "control: char budget alone should keep this in one chunk"
        assert len(capped) >= 3, "200 entries / 75 cap should split into at least 3 chunks"
        for c in capped:
            assert c.count("\n\n") + 1 <= 75 + 1, "no chunk exceeds the entry cap (+1 for overlap)"

    def test_max_entries_disabled_when_none(self):
        """``max_entries=None`` reproduces the legacy char-only behaviour."""
        from idi_corporate_structure.extractor import _chunk_document

        paragraphs = [f"Para{i}" for i in range(200)]
        text = "\n\n".join(paragraphs)

        chunks = _chunk_document(text, max_chars=10_000, overlap_chars=0, max_entries=None)
        assert len(chunks) == 1


class TestExtractWithChunking:
    """Tests for chunked extraction path in GptExtractor.extract()."""

    def test_no_chunking_for_small_doc(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result, _, _, num_chunks = extractor.extract(sample_filing, make_exhibit_response())

        assert num_chunks == 1
        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_preemptive_chunking_for_large_doc(self, sample_filing, mocker):
        """A document over the threshold is chunked preemptively (before any LLM call)."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            },
        )

        # Document well over the 4_000-char threshold, with paragraph breaks
        big_content = "Apple Operations LLC (Delaware)\n\n" + ("filler line\n\n" * 500)
        doc = make_exhibit_response(content=big_content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert num_chunks > 1
        # Even though every chunk returned the same sub, dedup keeps it once
        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_chunking_triggered_on_truncation(self, sample_filing, mocker):
        """A doc under the preemptive threshold but that one-shot-truncates falls back to chunking."""
        # Push the preemptive threshold high so this doc skips preemptive chunking,
        # but keep the per-chunk max smaller so the retry path produces multiple chunks.
        mocker.patch("idi_corporate_structure.extractor._CHUNK_THRESHOLD_CHARS", 100_000)
        mocker.patch("idi_corporate_structure.extractor._CHUNK_MAX_CHARS", 2_000)

        extractor = GptExtractor(openai_api_key="fake-key")
        call_log = []

        def fake_summarize(doc):
            call_log.append(len(doc))
            # First call (one-shot on full doc) raises truncated; chunked calls succeed.
            if len(call_log) == 1:
                raise ExtractionTruncatedError("test truncation")
            return {
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            }

        mocker.patch.object(extractor, "_summarize", side_effect=fake_summarize)

        # ~6K chars — under the patched 100K threshold but over the patched 2K chunk max.
        content = "Apple Operations LLC (Delaware)\n\n" + ("padding paragraph\n\n" * 400)
        doc = make_exhibit_response(content=content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert len(call_log) >= 2  # one-shot + at least one chunked call
        assert num_chunks > 1
        assert any(s.name == "Apple Operations LLC" for s in result)

    def test_chunked_truncation_bubbles_up(self, sample_filing, mocker):
        """If a chunk itself truncates, the error bubbles up — chunk size needs tuning."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor,
            "_summarize",
            side_effect=ExtractionTruncatedError("chunk too big"),
        )

        big_content = "x" * 10_000
        doc = make_exhibit_response(content=big_content)

        with pytest.raises(ExtractionTruncatedError):
            extractor.extract(sample_filing, doc)

    def test_chunked_dedupes_same_name_across_chunks(self, sample_filing, mocker):
        """Subsidiaries appearing in multiple chunks (via overlap) are deduped by name."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # Every chunk returns the same subsidiary
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "APPLE OPERATIONS LLC",  # different case → still same after _normalize
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                ]
            },
        )

        big_content = "Apple Operations LLC (Delaware)\n\n" + ("filler\n\n" * 500)
        doc = make_exhibit_response(content=big_content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert num_chunks > 1
        assert len([s for s in result if s.name.lower() == "apple operations llc"]) == 1


class TestPerChunkYieldLogging:
    """Per-chunk ``input rows → extracted`` logging in _summarize_chunks.

    The yield ratio is our cheapest signal for catching laziness regressions:
    if a chunk has 116 rows and the model returns 70, that's a 0.60 yield and
    deserves a WARNING in the run log.
    """

    def _build_chunked_doc(self, paragraphs: int = 300) -> dict:
        """Build a document large enough (in chars) to trigger chunking."""
        return make_exhibit_response(
            content="\n\n".join(
                f"Subsidiary Number {i:04d} Holdings LLC (Delaware)" for i in range(paragraphs)
            )
        )

    def test_yield_log_emitted_per_chunk(self, sample_filing, mocker):
        """One yield log line per chunk, at INFO level when yield is healthy."""
        extractor = GptExtractor(openai_api_key="fake-key")
        info_spy = mocker.spy(extractor._logger, "info")
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
                * 80
            },
        )

        _, _, _, num_chunks = extractor.extract(sample_filing, self._build_chunked_doc())

        yield_calls = [c for c in info_spy.call_args_list if "input rows" in c.args[0]]
        assert len(yield_calls) == num_chunks
        assert all("yield=" in c.args[0] for c in yield_calls)

    def test_low_yield_emits_warning(self, sample_filing, mocker):
        """Chunk yield below _LOW_YIELD_RATIO is logged as a WARNING."""
        extractor = GptExtractor(openai_api_key="fake-key")
        warning_spy = mocker.spy(extractor._logger, "warning")
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            },
        )

        extractor.extract(sample_filing, self._build_chunked_doc())

        yield_warnings = [c for c in warning_spy.call_args_list if "input rows" in c.args[0]]
        assert yield_warnings, "low-yield chunk should produce a WARNING-level yield log"

    def test_healthy_yield_does_not_warn(self, sample_filing, mocker):
        """High-yield chunks do not emit WARNING-level yield logs."""
        extractor = GptExtractor(openai_api_key="fake-key")
        warning_spy = mocker.spy(extractor._logger, "warning")

        def echo(doc):
            rows = [p for p in doc.split("\n\n") if p.strip()]
            return {
                "subsidiaries": [
                    {
                        "name": p.split(" (")[0],
                        "location": "Delaware",
                        "source_quote": p,
                    }
                    for p in rows
                ]
            }

        mocker.patch.object(extractor, "_summarize", side_effect=echo)

        extractor.extract(sample_filing, self._build_chunked_doc())

        yield_warnings = [c for c in warning_spy.call_args_list if "input rows" in c.args[0]]
        assert not yield_warnings, "healthy-yield chunks should not warn"


class TestRetryOutlierChunks:
    """Re-extraction of sibling-outlier low-yield chunks in _summarize_chunks."""

    @staticmethod
    def _sub(name: str) -> dict:
        return {"name": name, "location": "Delaware", "source_quote": name}

    @classmethod
    def _chunk(cls, prefix: str, n: int) -> str:
        return "\n\n".join(f"{prefix}{i} LLC (Delaware)" for i in range(n))

    def test_reextracts_sibling_outlier_chunk(self, mocker):
        """A non-final chunk whose yield is far below its siblings is re-extracted."""
        extractor = GptExtractor(openai_api_key="fake-key")
        chunks = [self._chunk("A", 10), self._chunk("B", 10), self._chunk("C", 10)]
        subs = [
            [self._sub(f"A{i} LLC") for i in range(9)],  # yield 0.9
            [self._sub(f"B{i} LLC") for i in range(3)],  # yield 0.3 → outlier (middle chunk)
            [self._sub(f"C{i} LLC") for i in range(9)],  # yield 0.9
        ]
        recover = mocker.patch.object(
            extractor,
            "_reextract_outlier",
            return_value=[self._sub(f"B{i} LLC") for i in range(10)],
        )

        extractor._retry_outlier_chunks(chunks, subs, "ACME", "url")

        recover.assert_called_once()
        assert [len(s) for s in subs] == [9, 10, 9]  # outlier recovered, siblings untouched

    def test_final_chunk_not_retried(self, mocker):
        """The last chunk is never retried: trailing notes make it a false outlier.

        Mirrors the observed Genie / DR Reddy's tail chunks (yield 0.48 / 0.32)
        that recovered nothing on retry.
        """
        extractor = GptExtractor(openai_api_key="fake-key")
        recover = mocker.patch.object(extractor, "_reextract_outlier")
        chunks = [self._chunk("A", 10), self._chunk("B", 10), self._chunk("C", 10)]
        subs = [
            [self._sub(f"A{i} LLC") for i in range(9)],  # yield 0.9
            [self._sub(f"B{i} LLC") for i in range(9)],  # yield 0.9
            [self._sub(f"C{i} LLC") for i in range(3)],  # yield 0.3 but it is the LAST chunk
        ]

        extractor._retry_outlier_chunks(chunks, subs, "Genie", "url")

        recover.assert_not_called()
        assert [len(s) for s in subs] == [9, 9, 3]

    def test_uniformly_low_yield_not_retried(self, mocker):
        """All-chunks-low (an input_rows artifact, e.g. VIASAT) must not trigger retries."""
        extractor = GptExtractor(openai_api_key="fake-key")
        recover = mocker.patch.object(extractor, "_reextract_outlier")
        chunks = [self._chunk(p, 100) for p in ("A", "B", "C")]
        subs = [[self._sub(f"{p}{i} LLC") for i in range(49)] for p in ("A", "B", "C")]  # ~0.49

        extractor._retry_outlier_chunks(chunks, subs, "VIASAT", "url")

        recover.assert_not_called()
        assert [len(s) for s in subs] == [49, 49, 49]

    def test_healthy_chunks_not_retried(self, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        recover = mocker.patch.object(extractor, "_reextract_outlier")
        chunks = [self._chunk(p, 10) for p in ("A", "B", "C")]
        subs = [[self._sub(f"{p}{i} LLC") for i in range(10)] for p in ("A", "B", "C")]  # 1.0

        extractor._retry_outlier_chunks(chunks, subs, "ACME", "url")

        recover.assert_not_called()

    def test_single_chunk_never_retried(self, mocker):
        """With no siblings there is no baseline to compare against."""
        extractor = GptExtractor(openai_api_key="fake-key")
        recover = mocker.patch.object(extractor, "_reextract_outlier")

        extractor._retry_outlier_chunks([self._chunk("A", 10)], [[self._sub("A0 LLC")]], "X", "url")

        recover.assert_not_called()

    def test_reextract_outlier_splits_into_smaller_subchunks(self, mocker):
        """_reextract_outlier re-summarizes the chunk in smaller entry-capped pieces."""
        extractor = GptExtractor(openai_api_key="fake-key")
        calls = []

        def fake(doc):
            calls.append(doc)
            rows = [p for p in doc.split("\n\n") if p.strip()]
            return {"subsidiaries": [self._sub(p.split(" (")[0]) for p in rows]}

        mocker.patch.object(extractor, "_summarize", side_effect=fake)

        recovered = extractor._reextract_outlier(self._chunk("Row", 60))  # 60 > _RETRY_MAX_ENTRIES

        assert len(calls) >= 2  # split into multiple smaller calls
        assert len({s["name"] for s in recovered}) == 60  # every distinct row recovered

    def test_reextract_outlier_swallows_truncation(self, mocker):
        """A truncating sub-chunk is skipped, not raised — retry is best-effort."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(extractor, "_summarize", side_effect=ExtractionTruncatedError("boom"))

        assert extractor._reextract_outlier(self._chunk("Row", 60)) == []

    def test_reextract_outlier_swallows_timeout(self, mocker):
        """A timed-out sub-chunk must not sink an already-successful extraction."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor, "_summarize", side_effect=ExtractionTimeoutError("timed out")
        )

        assert extractor._reextract_outlier(self._chunk("Row", 60)) == []

    def test_reextract_outlier_swallows_runtime_error(self, mocker):
        """A generic API RuntimeError on the best-effort retry is logged and skipped."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(extractor, "_summarize", side_effect=RuntimeError("api error"))

        assert extractor._reextract_outlier(self._chunk("Row", 60)) == []


class TestWindowedSubsequenceGrounding:
    """Windowed in-order token fallback + control/zero-width stripping in grounding."""

    def test_recovers_table_shattered_name(self):
        """A name whose words are split across table columns is still grounded."""
        from idi_corporate_structure.extractor import _is_name_in_document

        # Name cell wrapped across rows, interleaved with the row's other columns.
        doc = "PT Telekomunikasi ​ Mobile ​ 1995 ​ 70 ​ 70 selular ​ telecommunication"
        assert _is_name_in_document("PT Telekomunikasi Selular", doc)

    def test_rejects_tokens_scattered_beyond_window(self):
        """Tokens present but far apart do not ground — the hallucination guard holds."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Alpha " + ("filler " * 80) + "Beta"  # ~560 chars apart, > 250-char window
        assert not _is_name_in_document("Alpha Beta", doc)

    def test_single_token_not_resurrected_by_window(self):
        from idi_corporate_structure.extractor import _is_name_in_document

        assert not _is_name_in_document("Ghost", "Alpha Beta Gamma Holdings")

    def test_absent_later_token_rejected_despite_repeated_first_token(self):
        """A missing later token grounds to False even when the first token recurs.

        The early exit must not be defeated by extra occurrences of the first token.
        """
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Alpha Corp and Alpha Trust and Alpha Group"  # "Alpha" thrice, "Beta" absent
        assert not _is_name_in_document("Alpha Beta", doc)

    def test_windowed_match_logs_debug(self, mocker):
        from idi_corporate_structure import extractor as ext_mod

        mock_logger = mocker.patch.object(ext_mod, "get_logger")
        doc = "PT Telekomunikasi ​ Mobile ​ 1995 selular ​ telecommunication"

        ext_mod._is_name_in_document("PT Telekomunikasi Selular", doc)

        assert mock_logger.return_value.debug.called
        assert "windowed-subsequence" in mock_logger.return_value.debug.call_args[0][0]

    def test_zero_width_space_stripped_in_grounding(self):
        from idi_corporate_structure.extractor import _is_name_in_document

        assert _is_name_in_document("FooBar Holdings LLC", "Foo​Bar Holdings LLC (Delaware)")

    def test_control_char_stripped_in_grounding(self):
        """A C0 control char embedded in both name and doc is normalized away consistently."""
        from idi_corporate_structure.extractor import _is_name_in_document

        assert _is_name_in_document(
            "USCNHK Group\x01Limited", "USCNHK Group\x01Limited (Hong Kong)"
        )


class TestIsNameGrounded:
    """The pre-normalized grounding core shared across a document's names."""

    def test_matches_each_grounding_tier_with_prenormalized_forms(self):
        from idi_corporate_structure.extractor import (
            _compact,
            _is_name_grounded,
            _normalize,
        )

        doc = "Apple Operations LLC (Delaware) — PT Telekomunikasi ​ Mobile selular"
        doc_norm, doc_compact = _normalize(doc), _compact(doc)

        assert _is_name_grounded("Apple Operations LLC", doc_norm, doc_compact)  # strict
        assert _is_name_grounded("Apple Operations, LLC.", doc_norm, doc_compact)  # compact
        assert _is_name_grounded("PT Telekomunikasi Selular", doc_norm, doc_compact)  # windowed
        assert not _is_name_grounded("Ghost Corp", doc_norm, doc_compact)
        assert not _is_name_grounded("", doc_norm, doc_compact)

    def test_wrapper_matches_core(self):
        """_is_name_in_document is a thin wrapper that agrees with _is_name_grounded."""
        from idi_corporate_structure.extractor import (
            _compact,
            _is_name_grounded,
            _is_name_in_document,
            _normalize,
        )

        doc = "Johnson & Johnson (Singapore) HoldCo LLC (Delaware)"
        name = "Johnson & Johnson Singapore HoldCo LLC"

        assert _is_name_in_document(name, doc) == _is_name_grounded(
            name, _normalize(doc), _compact(doc)
        )

    def test_document_not_renormalized_per_name(self, mocker):
        """The document is normalized a constant number of times, not once per name."""
        from idi_corporate_structure import extractor as ext_mod

        extractor = GptExtractor(openai_api_key="fake-key")
        normalize_spy = mocker.spy(ext_mod, "_normalize")
        compact_spy = mocker.spy(ext_mod, "_compact")

        document = make_exhibit_response(
            content="Apple Operations LLC (Delaware)\n\nApple Europe Ltd (Ireland)"
        )
        doc_text = document["data"]
        subs = [
            {"name": "Apple Operations LLC", "location": "Delaware", "source_quote": ""}
            for _ in range(5)
        ]

        extractor._locate_grounded_subsidiaries(subs, document)

        # Were the document re-normalized per name, these would be ~5x higher. The doc
        # is normalized once directly + once inside the single _compact(doc) call.
        assert normalize_spy.call_args_list.count(mocker.call(doc_text)) <= 2
        assert compact_spy.call_args_list.count(mocker.call(doc_text)) == 1


class TestJnjChunkingFixture:
    """End-to-end chunking test against the real Johnson & Johnson Exhibit 21.

    The fixture (``tests/processor/fixtures/jnj_ex21_subsidiaries.htm``) is the
    same exhibit that empirically caused gpt-4.1-nano to give up at ~70 of 410
    subsidiaries in production. This test mocks the OpenAI call so no API
    request is made; it verifies that:
      * the exhibit triggers preemptive chunking (>1 chunk),
      * one ``_summarize`` call is made per chunk,
      * subsidiaries returned by every chunk make it into the final result,
      * dedup collapses overlap-region duplicates.
    """

    def _load_jnj_document(self) -> dict:
        """Build a document dict from the J&J fixture (HTML → plain text)."""
        raw_html = _JNJ_FIXTURE.read_text(encoding="utf-8")
        return {
            "url": "https://www.sec.gov/Archives/edgar/data/200406/000020040625000038/ex21-subsidiariesxform10xk.htm",
            "data": html_to_text(raw_html),
        }

    def test_jnj_exhibit_is_chunked(self, sample_filing, mocker):
        """The J&J exhibit is well over the chunking threshold and should be split."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # No-op model: returns nothing per chunk. We only care about the chunk count here.
        mocker.patch.object(extractor, "_summarize", return_value={"subsidiaries": []})

        _, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert num_chunks > 1, "J&J exhibit should trigger chunking"

    def test_jnj_chunking_calls_summarize_once_per_chunk(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        spy_summarize = mocker.patch.object(
            extractor, "_summarize", return_value={"subsidiaries": []}
        )

        _, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert spy_summarize.call_count == num_chunks

    def test_jnj_chunking_makes_no_real_api_call(self, sample_filing, mocker):
        """Mocking _summarize means the underlying HTTP client is never touched."""
        extractor = GptExtractor(openai_api_key="fake-key")
        spy_query = mocker.spy(extractor._openai_client, "query_endpoint")
        mocker.patch.object(extractor, "_summarize", return_value={"subsidiaries": []})

        extractor.extract(sample_filing, self._load_jnj_document())

        assert spy_query.call_count == 0

    def test_jnj_chunking_aggregates_results_across_chunks(self, sample_filing, mocker):
        """Each chunk returns its own subsidiary; merged result includes all."""
        # Real subsidiary names from the J&J Exhibit 21 — chosen so they ground.
        names_per_chunk = [
            "ABD Holding Company, Inc.",
            "DePuy Synthes, Inc.",
            "Janssen Biotech, Inc.",
            "Mentor Worldwide LLC",
            "Synthes USA, LLC",
        ]
        extractor = GptExtractor(openai_api_key="fake-key")

        call_idx = {"i": 0}

        def fake_summarize(_chunk: str) -> dict:
            i = call_idx["i"]
            call_idx["i"] += 1
            if i < len(names_per_chunk):
                return {
                    "subsidiaries": [
                        {
                            "name": names_per_chunk[i],
                            "location": "Delaware",
                            "source_quote": names_per_chunk[i],
                        }
                    ]
                }
            return {"subsidiaries": []}

        mocker.patch.object(extractor, "_summarize", side_effect=fake_summarize)

        result, ungrounded_name, _, num_chunks = extractor.extract(
            sample_filing, self._load_jnj_document()
        )

        assert num_chunks >= len(names_per_chunk)
        assert ungrounded_name == 0, "All sample names should ground against the real exhibit"
        result_names = {s.name for s in result}
        for name in names_per_chunk:
            assert name in result_names

    def test_jnj_chunking_dedupes_overlap_duplicates(self, sample_filing, mocker):
        """A subsidiary returned by every chunk (overlap simulation) appears once."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # Same sub returned by every call — dedup-by-name should collapse to one row.
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "ABD Holding Company, Inc.",
                        "location": "Delaware",
                        "source_quote": "ABD Holding Company, Inc.",
                    }
                ]
            },
        )

        result, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert num_chunks > 1
        assert len([s for s in result if s.name == "ABD Holding Company, Inc."]) == 1


class TestHtmlToText:
    """Tests for html_to_text — ensuring table cells are space-delimited."""

    def test_two_column_table_row_uses_space_separator(self):
        """Each <td> cell is separated by a space."""
        html = "<table><tr><td>Apsis</td><td>France</td></tr></table>"
        result = html_to_text(html)
        assert result == "Apsis France"

    def test_multi_word_name_and_single_word_location(self):
        r"""Multi-word names joined by \xa0 in the source are preserved."""
        html = "<table><tr><td>AMO\xa0Uppsala\xa0AB</td><td>Sweden</td></tr></table>"
        result = html_to_text(html)
        assert result == "AMO\xa0Uppsala\xa0AB Sweden"

    def test_name_with_abbreviation_period_and_location(self):
        """Names ending in 'S.L.' are preserved with a space before the location."""
        html = "<table><tr><td>Cilag-Biotech,&#160;S.L.</td><td>Spain</td></tr></table>"
        result = html_to_text(html)
        assert result == "Cilag-Biotech,\xa0S.L. Spain"

    def test_multi_word_location(self):
        """Multi-word locations (e.g. 'Cayman Islands') are kept intact after the space."""
        html = "<table><tr><td>AMO\xa0Ireland</td><td>Cayman\xa0Islands</td></tr></table>"
        result = html_to_text(html)
        assert result == "AMO\xa0Ireland Cayman\xa0Islands"

    def test_multiple_rows_each_on_own_line(self):
        """Multiple rows produce one space-delimited entry per line."""
        html = (
            "<table>"
            "<tr><td>Alpha Corp</td><td>Delaware</td></tr>"
            "<tr><td>Beta\xa0Inc.</td><td>Ireland</td></tr>"
            "</table>"
        )
        lines = [line for line in html_to_text(html).splitlines() if line.strip()]
        assert lines[0] == "Alpha Corp Delaware"
        assert lines[1] == "Beta\xa0Inc. Ireland"

    def test_three_column_table_space_delimited(self):
        """Three-column rows produce two spaces between columns."""
        html = "<table><tr><td>GEAE Technology, Inc.</td><td>100</td><td>Delaware</td></tr></table>"
        result = html_to_text(html)
        assert result == "GEAE Technology, Inc. 100 Delaware"

    def test_non_table_content_unchanged(self):
        """Plain paragraphs without table structure are not affected."""
        html = "<p>Some plain text here.</p>"
        result = html_to_text(html)
        assert result == "Some plain text here."


class TestCleanName:
    """Tests for _clean_name — invisible-character stripping and NBSP normalisation."""

    def test_nbsp_replaced_with_space(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("ASM\xa0Services") == "ASM Services"

    def test_zwnj_stripped(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("Silex Spain, S.L.\u200c") == "Silex Spain, S.L."

    def test_zwsp_stripped(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("Golden\u200bMinerals") == "GoldenMinerals"

    def test_zwj_stripped(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("Corp\u200d Ltd") == "Corp Ltd"

    def test_bom_stripped(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("\ufeffHoldings Inc.") == "Holdings Inc."

    def test_html_entities_decoded(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("Johnson &amp; Johnson") == "Johnson & Johnson"

    def test_double_spaces_collapsed(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("A\xa0 B") == "A B"

    def test_real_world_golden_minerals_example(self):
        from idi_corporate_structure.extractor import _clean_name

        dirty = "ASM Services\xa0S.a\xa0r.l.\u200c"
        assert _clean_name(dirty) == "ASM Services S.a r.l."

    def test_trailing_footnote_marker_stripped(self):
        """A footnote ref appended straight onto a name must not leak into storage.

        E.g. 'Freedom Bank Kazakhstan JSC, Kazakhstan(1)' \u2014 grounding has
        already run by the time this cleanup applies, so it's safe to strip.
        """
        from idi_corporate_structure.extractor import _clean_name

        assert (
            _clean_name("Freedom Bank Kazakhstan JSC, Kazakhstan(1)")
            == "Freedom Bank Kazakhstan JSC, Kazakhstan"
        )

    def test_trailing_footnote_marker_with_space_stripped(self):
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("Example Corp (2)") == "Example Corp"

    def test_parenthetical_not_at_end_preserved(self):
        """A parenthetical that's part of the legal name must not be stripped.

        Only a trailing footnote digit in parens should be, e.g. "(1)".
        """
        from idi_corporate_structure.extractor import _clean_name

        assert _clean_name("U-Haul Co. (Canada) Ltd.") == "U-Haul Co. (Canada) Ltd."


class TestCompactGroundingFallback:
    """Tests for _compact and the compact fallback in _is_name_in_document."""

    def test_compact_lowercases_and_strips_punctuation(self):
        from idi_corporate_structure.extractor import _compact

        assert (
            _compact("Johnson & Johnson (Singapore) HoldCo LLC")
            == "johnsonjohnsonsingaporeholdcollc"
        )

    def test_strict_match_still_works(self):
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "ABD Holding Company, Inc. (Delaware)"
        assert _is_name_in_document("ABD Holding Company, Inc.", doc)

    def test_jnj_singapore_matches_via_fallback(self):
        """Model dropped the parens around 'Singapore' — compact fallback recovers it."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Johnson & Johnson (Singapore) HoldCo LLC, a Delaware corporation"
        assert _is_name_in_document("Johnson & Johnson Singapore HoldCo LLC", doc)

    def test_jnj_healthcare_matches_via_fallback(self):
        """Model merged 'Health Care' into 'Healthcare' — compact fallback recovers it."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Johnson & Johnson Health Care Systems Inc. (New Jersey)"
        assert _is_name_in_document("Johnson & Johnson Healthcare Systems Inc.", doc)

    def test_empty_name_returns_false(self):
        from idi_corporate_structure.extractor import _is_name_in_document

        assert not _is_name_in_document("", "some document text")

    def test_hallucinated_name_still_returns_false(self):
        """A name with no compact substring match is correctly rejected."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Johnson & Johnson Health Care Systems Inc. (New Jersey)"
        assert not _is_name_in_document("Completely Made Up Pharma Corp", doc)

    def test_no_false_match_on_unrelated_name(self):
        """A name with no compact substring match is correctly rejected."""
        from idi_corporate_structure.extractor import _is_name_in_document

        doc = "Pineapple Holdings Ltd (California)"
        assert not _is_name_in_document("Acme Corporation", doc)

    def test_compact_fallback_logs_debug(self, mocker):
        """When the compact path fires, a DEBUG message is emitted."""
        from idi_corporate_structure import extractor as ext_mod

        mock_logger = mocker.patch.object(ext_mod, "get_logger")
        mock_log_instance = mock_logger.return_value

        doc = "Johnson & Johnson (Singapore) HoldCo LLC, a Delaware corporation"
        ext_mod._is_name_in_document("Johnson & Johnson Singapore HoldCo LLC", doc)

        assert mock_log_instance.debug.called
        call_args = mock_log_instance.debug.call_args[0]
        assert "compact fallback" in call_args[0]

    def test_location_check_skipped_when_name_only_matched_via_compact_fallback(self, mocker):
        """A name only grounded via the compact fallback has no exact doc position.

        Its normalized form's ``.find()`` returns -1. The proximity window
        used to be built from that -1 as if it were a real index — silently
        checking the start of the document instead of near the name. It
        should skip the check instead.
        """
        from idi_corporate_structure.extractor import GptExtractor, _normalize

        extractor = GptExtractor(openai_api_key="fake-key")
        debug_spy = mocker.spy(extractor._logger, "debug")
        doc = "Johnson & Johnson (Singapore) HoldCo LLC, a Delaware corporation"

        ungrounded = extractor._is_location_grounded(
            name="Johnson & Johnson Singapore HoldCo LLC",  # only matches via compact fallback
            location="Delaware",
            doc_text_normalized=_normalize(doc),
            doc_url="https://example.com",
        )

        assert ungrounded == 0
        assert any(
            "Skipping location-proximity check" in str(call.args[0])
            for call in debug_spy.call_args_list
        )


class TestCleanNameWiredIntoExtract:
    """End-to-end tests verifying _clean_name runs on every stored Subsidiary.name."""

    def test_dirty_name_stored_clean(self, sample_filing, mocker):
        """Invisible chars and NBSPs in the model's name are cleaned before storing."""
        extractor = GptExtractor(openai_api_key="fake-key")
        dirty_name = "ASM\xa0Services\u200c"
        doc_content = "ASM Services (Delaware)"
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": dirty_name, "location": "Delaware", "source_quote": "ASM Services"}]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response(content=doc_content))

        assert len(result) == 1
        assert result[0].name == "ASM Services"

    def test_compact_fallback_allows_grounding_and_name_stored(self, sample_filing, mocker):
        """When the model's name differs by parens/spacing, compact fallback grounds it."""
        extractor = GptExtractor(openai_api_key="fake-key")
        doc_content = "Johnson & Johnson (Singapore) HoldCo LLC, a Delaware corporation"
        model_name = "Johnson & Johnson Singapore HoldCo LLC"
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": model_name, "location": "Delaware", "source_quote": model_name}]
            ),
        )
        result, ungrounded, *_ = extractor.extract(
            sample_filing, make_exhibit_response(content=doc_content)
        )

        assert ungrounded == 0, "Compact fallback should have accepted the name"
        assert len(result) == 1
        assert result[0].name == model_name

    def test_compact_fallback_name_does_not_corrupt_location_grounding_count(
        self, sample_filing, mocker
    ):
        """A compact-fallback-only name must not corrupt the ungrounded_location count.

        See test_location_check_skipped_when_name_only_matched_via_compact_fallback
        for the bogus-window bug this guards against. It should stay 0.
        """
        extractor = GptExtractor(openai_api_key="fake-key")
        doc_content = "Johnson & Johnson (Singapore) HoldCo LLC, a Delaware corporation"
        model_name = "Johnson & Johnson Singapore HoldCo LLC"
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": model_name, "location": "Delaware", "source_quote": model_name}]
            ),
        )

        result, ungrounded_name, ungrounded_location, _ = extractor.extract(
            sample_filing, make_exhibit_response(content=doc_content)
        )

        assert ungrounded_name == 0
        assert ungrounded_location == 0
        assert len(result) == 1


class TestTakeOverlap:
    """Tests for _take_overlap (paragraph-aligned chunk overlap)."""

    def test_returns_empty_when_overlap_is_zero(self):
        from idi_corporate_structure.extractor import _take_overlap

        assert _take_overlap("anything\n\nelse", overlap_chars=0) == ""

    def test_returns_only_whole_paragraphs(self):
        """Overlap never includes a partial paragraph, even if a whole one exceeds the budget."""
        from idi_corporate_structure.extractor import _take_overlap

        text = "para1\n\npara2\n\nlongest_paragraph_with_lots_of_content"
        overlap = _take_overlap(text, overlap_chars=20)

        # The final paragraph alone exceeds 20 chars but must still be returned whole.
        assert overlap == "longest_paragraph_with_lots_of_content"

    def test_takes_multiple_paragraphs_within_budget(self):
        from idi_corporate_structure.extractor import _take_overlap

        text = "para1\n\npara2\n\npara3\n\npara4"
        overlap = _take_overlap(text, overlap_chars=100)

        # All paragraphs fit within 100 chars, so all are kept.
        assert overlap == text

    def test_no_partial_word_at_overlap_boundary(self):
        """Regression test for the Transco 'Eight Project LLC' bug: overlap never bisects a name."""
        from idi_corporate_structure.extractor import _take_overlap

        text = (
            "Perryville Gas Storage LLC Delaware\n\n"
            "Power Eight Project LLC Delaware\n\n"
            "Reserveco Inc. Delaware"
        )
        overlap = _take_overlap(text, overlap_chars=30)

        # Whatever fits, it must be the whole final paragraph — never a slice of "Power Eight Project LLC".
        assert "wer Eight Project LLC" not in overlap
        assert overlap.startswith("Power Eight Project LLC") or overlap == "Reserveco Inc. Delaware"
