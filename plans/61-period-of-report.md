# Add `period_of_report` to the corporate-structure output

Issue: https://github.com/dsi-clinic/idi-corporate-structure/issues/61

> **Note on the code this plan targets.** It is written against `origin/dev`
> (`a220207`), which is well ahead of the local checkout. PRs
> [#50](https://github.com/dsi-clinic/idi-corporate-structure/pull/50) and
> [#57](https://github.com/dsi-clinic/idi-corporate-structure/pull/57) replaced
> the `submissions.zip` input path entirely. Sync before starting; `pyproject`
> now requires `idi-ftm2j-shared>=0.1.16` (the local venv still has 0.1.3).

## Goal

Carry each filing's SEC reporting period into the output parquet as a
`period_of_report` column, so downstream consumers can select the correct
fiscal year instead of guessing from `filing_date` + `accession_number`.

## Background — what was verified

**One source covers both the bulk historical run and the daily routine run,
because the split lives upstream in `idi-sec-scraper` and converges before this
repo sees anything.**

- `HistoricalDiscovery` (`idi_sec_scraper/discovery.py:80`) reads
  `submissions.zip`; `DailyDiscovery` (`discovery.py:55`) reads the daily
  crawler index. Both yield `DiscoveredFiling`.
- Both then run through the *same* `SECScraperPipeline._scrape_filing_inner`,
  which calls `parse_index_htm` and assigns `scraped_filing.report_date`
  (`idi_sec_scraper/pipeline.py:354`). `HistoricalSECScraperPipeline` and
  `DailySECScraperPipeline` override only `load_input` / `_make_discovery`.
- `idi-corporate-structure` reads neither. `PipelineConfig` has no
  `input_file`; it takes `start_date` / `end_date` / `sec_bucket`, and
  `load_input` calls `iter_filings_by_form_type(..., search_by="scraped_date")`
  for every run. Historical coverage here is a wide date range over
  backfilled scrapes, not a zip.

The `submissions.zip` curator scripts still in `scripts/` (`curate_submissions.py`,
`curate_submissions_by_accession.py`, `select_ground_truth.py`) reference a
`config.input_file` that no longer exists on this pipeline. They are orphaned
dev utilities, not a live input path — see "Out of scope".

So a single source of the reporting period serves both cases:

1. **`ScrapedFiling.report_date` already exists and is already populated.**
   The shared library's `ScrapedFiling` dataclass carries
   `report_date: str = ""`, deserialised from each filing's `manifest.json`.
   `idi-sec-scraper` fills it in `parse_index_htm`
   (`src/idi_sec_scraper/parser.py:45`) from the **Period of Report** field on
   the EDGAR filing index page, normalised through `datetime.date.fromisoformat`
   and written as `.isoformat()` — so it is already ISO `YYYY-MM-DD`, or `""`
   when the index page has no such field. `git log -S"scraped_filing.report_date"`
   returns exactly one commit — `fc5dda8`, the scraper's initial implementation
   (2026-05-06) — so **no manifest predates the field**, on either the
   historical or the daily path.

2. **It resolves the co-registrant case.** Fetching the two CIK 1583994 index
   pages and running the scraper's exact extraction path over them:

   | accession | filingDate | Period of Report |
   |---|---|---|
   | `0001583994-17-000009` | 2017-02-24 | **2014-12-31** |
   | `0001574540-17-000007` | 2017-02-24 | **2016-12-31** |

   The index page carries one Period of Report per accession, so the
   co-registrant filing under CIK 1574540's archive folder is dated correctly.

3. **The submissions JSON is a free fallback.**
   `_fetch_company_meta` already fetches
   `https://data.sec.gov/submissions/CIK{cik}.json` once per CIK and caches it.
   That same response contains `filings.recent.reportDate` parallel to
   `accessionNumber` — verified to agree with the index pages above, and to
   give CIK 857501's three same-day 2017-07-03 filings `2014-05-31` /
   `2015-05-31` / `2016-05-31` (note: fiscal year ends **May 31**). So a
   fallback for a blank `report_date` costs no additional request.

Two corrections to the issue text:

- The issue says the value is "already in the submission header the processor
  reads to locate `exhibit_url`." The processor does not read a submission
  header — exhibit URLs now come from the scraper's `manifest.json` documents.
  This makes the change *cheaper* than described: the period is already sitting
  on the object `load_input` iterates.
- AC #3 does not hold as written (see "Restated acceptance criteria").

## Confirmed requirements

- `Filing` and `Subsidiary` gain a `period_of_report` field, sourced from
  `ScrapedFiling.report_date`, with the submissions JSON as fallback.
- The output parquet gains a `period_of_report` column, non-null for every row.
- Rows already in the parquet are backfilled without re-running extraction.
- A filing with no period from either source is dropped rather than written
  with a null.
- `README.md` documents the new column and clarifies `filing_date`.

## Restated acceptance criteria

AC #3 in the issue is not satisfiable as written and is replaced by #3a/#3b.
CIK 857501 also filed a 10-K for period `2017-05-31` on 2018-06-27, so the max
period for that CIK is FY2017, not FY2016; and its 10-K/A `0001065949-17-000093`
(filed 2017-07-05) *ties* the FY2016 10-K at `2016-05-31`, so max-period alone
does not disambiguate 10-K from 10-K/A. Defining that tie-break is downstream
selection logic and is out of scope here.

1. Output carries a `period_of_report` column as an ISO `YYYY-MM-DD` string,
   non-null and non-empty for every row.
2. Values match EDGAR's `reportDate` for the same `accession_number`, including
   co-registrant filings whose accession lives under another CIK's archive
   folder. Verified on CIK 1583994: `0001583994-17-000009` → `2014-12-31`,
   `0001574540-17-000007` → `2016-12-31`.
3. **a.** For CIK 1583994, selecting the row set with the max `period_of_report`
   yields the `0001574540-17-000007` (FY2016) exhibit — 261 subsidiaries, not
   the 94-subsidiary FY2014 list that the higher accession number would pick.
   **b.** For CIK 857501, the three accessions filed 2017-07-03 carry
   `2014-05-31`, `2015-05-31`, `2016-05-31` respectively; within that
   `filing_date`, max `period_of_report` selects `0001065949-17-000087` (FY2016).
4. Every row written by a new pipeline run has a non-empty `period_of_report`.
5. Rows written by earlier runs are filled by a one-off backfill script (see
   "Backfill" below) — this, not the pipeline, is what makes AC #1 true today.
6. Filings with no period from either source are not written to the output and
   are recorded in `failures.json` as `missing_period_of_report`.
7. `README.md` lists the new column in the output-layout block and notes that
   `filing_date` is the submission date, not the reporting period.

## Approach

### 1. Thread `report_date` onto `Filing`

- [`Filing`](src/idi_corporate_structure/types.py): add `period_of_report: str`.
- In `load_input`, set `period_of_report=scraped_filing.report_date` on the
  `Filing` constructed from each `ScrapedFiling`.

That is the entire happy path — no parsing, no new requests, no format
conversion.

### 2. Fall back to the submissions JSON when `report_date` is blank

Older manifests, or index pages without a Period of Report field, yield `""`.
Rather than dropping those filings immediately, reuse the response
`_fetch_company_meta` already fetches:

- Extend the per-CIK cache to also memoise an `accession_number → reportDate`
  dict built from `filings.recent` (and any `filings.files` overflow entries
  already present in the response) in the same pass that builds `CompanyMeta`.
  Zero extra requests; the CIK JSON is fetched and cached either way.
- When `scraped_filing.report_date` is empty, look the accession up in that map.

This was not in the original decision (fail immediately on blank), but it is
free and strictly reduces dropped filings, so it is folded in ahead of the drop
path rather than replacing it.

### 3. Drop filings with no period from either source

- Add `MISSING_PERIOD_OF_REPORT = "missing_period_of_report"` to
  [`FailureType`](src/idi_corporate_structure/failures.py) and to the
  classifier's `_DO_NOT_RETRY` set — if neither EDGAR surface dates the filing,
  retrying will not change that.
- In `load_input`, after the fallback, skip any filing still lacking a period:
  `_record_failure((cik, accession_number), ..., stat_keys=("failed_filings",))`,
  matching the existing `NO_EXHIBIT_FOUND` branch exactly. The registry is
  already keyed on `(cik, accession_number)` and `_should_skip` already consults
  it, so the skip persists correctly across runs with no new plumbing.

### 4. Carry it onto each extracted row

- [`Subsidiary`](src/idi_corporate_structure/types.py): add
  `period_of_report: str` (after `filing_date`; parquet column order follows
  dataclass field order).
- `extractor.py:600` — add `period_of_report=filing.period_of_report` to the
  `Subsidiary(...)` construction.

### 5. Backfill existing parquet rows

**The originally chosen in-pipeline auto-fill does not work under the current
input model, and this is the one place the plan departs from the earlier
decision.** `load_input` only ever sees filings inside the run's
`scraped_date` window — for a daily run, that is one day. Rows written by
previous runs are never revisited, so a `save_output` join could not reach them
and AC #1 would stay false indefinitely.

Two pieces instead:

- **New rows** get the value directly from step 1. No join, no backfill logic in
  `save_output` at all — simpler than the original plan.
- **Existing rows**: `scripts/backfill_period_of_report.py`. Reads the output
  parquet, takes the distinct `parent_cik` set, fetches
  `https://data.sec.gov/submissions/CIK{cik}.json` once per CIK through
  `SecClient` (rate-limited), builds the accession → `reportDate` map, and
  fills every row whose `period_of_report` is null or empty. Idempotent, safe to
  re-run, `--dry-run` to report the fill count and any unresolved accessions
  before writing. This is the same lookup as step 2, so factor it into one
  helper both call.

`save_output` needs one small change only: `pd.concat` of an older parquet with
new rows leaves `NaN` in `period_of_report` for the old rows, so add the column
if absent and normalise `NaN` → `""` before write, to keep the column's dtype
stable across runs.

### 6. Docs

- `README.md:19` — add `period_of_report` to the output-layout column list
  (note the output file is now `latest.parquet` per PR #57).
- Pipeline overview stage 1 — mention the reporting period alongside filing date.
- Add a short note that `filing_date` is the submission date and
  `period_of_report` is the fiscal period the exhibit describes, with the
  delinquent-filer case as the one-line motivation.

### 7. Tests

- `tests/conftest.py`: add `report_date` to the `ScrapedFiling` fixtures/builders
  and `period_of_report` to the `Filing` fixtures.
- `tests/processor/test_pipeline.py`:
  - `load_input` copies `report_date` onto `Filing.period_of_report`.
  - A blank `report_date` falls back to the submissions-JSON map and the filing
    survives.
  - Blank in both sources → filing dropped, `missing_period_of_report`
    registered under `(cik, accession_number)`, `failed_filings` incremented.
  - The fallback issues no extra SEC request beyond the one
    `_fetch_company_meta` already makes (assert on the mock's call count).
  - `save_output` writes `""` rather than `NaN` when merging a legacy parquet
    that lacks the column.
- `tests/processor/test_types.py`: `Filing` and `Subsidiary` field assertions.
- `tests/processor/test_extractor.py`: extracted rows carry `period_of_report`.
- New `tests/scripts/test_backfill_period_of_report.py`: fills blanks, leaves
  populated values alone, is idempotent, and reports unresolved accessions.

### 8. Verification of AC #2/#3

Add a `period` check to `scripts/verify_output.py`: assert the column is
non-empty everywhere, and spot-check CIK 1583994 and CIK 857501 against
`https://data.sec.gov/submissions/CIK{cik}.json`.

**Caveat:** `scripts/verify_output.py` does not currently run — on `origin/dev`
it still imports `idi_corporate_structure.common.api` and
`idi_corporate_structure.processor.extractor`, neither of which exists in the
flat `src/idi_corporate_structure/` layout. Repairing those imports is a
prerequisite; if deferred, verify AC #2/#3 with a one-off script and drop this
step.

## Assumptions

1. **Source is `ScrapedFiling.report_date`**, already on the object `load_input`
   iterates, with the submissions JSON as a no-cost fallback. Overrides the
   issue's description of where the value lives.
2. **Stored as a `str` in `YYYY-MM-DD` form**, matching `filing_date`. Both
   sources already emit that format. Not a pandas datetime dtype — that would
   make `period_of_report` inconsistent with its sibling date column.
3. **Scraped manifests reliably carry `report_date`, on both the historical and
   daily paths.** The field has been written by the shared
   `_scrape_filing_inner` since the scraper's first commit, so no manifest
   predates it and the fallback in step 2 should be near-dead code. Filings that
   fail parse validation never reach this pipeline at all — `manifest.parquet`
   indexes only successfully-scraped documents. **Confirmed against real
   manifests: `report_date` is present and in `YYYY-MM-DD` form.**
4. **Dropping undated filings is acceptable**, and after the fallback should be
   close to zero volume.
5. **No output-schema versioning exists** in this repo, so a new column is an
   additive change.

## Out of scope

- **Downstream "latest filing" selection logic.** Defining the max-period rule
  and its tie-break (10-K vs 10-K/A at the same period) belongs to the consumer.
  Worth a follow-up issue — AC #3's original wording shows the rule is not yet
  pinned down.
- **Adding `report_date` to the shared library's `manifest.parquet` query
  index** (`_MANIFEST_QUERY_COLUMNS`). Not needed: `_load_filing` reads the full
  `manifest.json`, which has it. Would only matter if a consumer wanted to
  *filter* by period at query time.
- **Repairing `scripts/verify_output.py`'s stale imports** beyond what step 8
  needs.
- **Removing or rewiring the orphaned `submissions.zip` curator scripts.** They
  target an input path the pipeline no longer has. Worth a follow-up issue, but
  unrelated to this column.
- **Reprocessing or re-extracting** any filing. No OpenAI calls are re-paid.

## Risks and open questions

- ~~The step-2/step-5 fallback is unverified against real manifests.~~
  **Resolved.** Confirmed that `manifest.json` tracks `report_date` in
  `YYYY-MM-DD` form, so step 1 is the real path and step 2 is a safety net for
  index pages that carry no Period of Report field. The residual unknown is only
  the rate of those blanks, which the `missing_period_of_report` counter will
  surface on the first run — not a reason to hold the build.
- **Backfill coverage is unmeasurable from here.** The output parquet is not
  present locally, so the number of rows needing backfill is unknown. Run the
  script's `--dry-run` first.
- **Column ordering churn.** Inserting `period_of_report` mid-dataclass changes
  parquet column order. Harmless for `pd.read_parquet` consumers; no positional
  consumer is known in this repo.
- **Open:** should a nonzero `missing_period_of_report` count fail the run, or
  just be reported in `display_stats`? Defaulting to report-only, consistent
  with every other failure type.
