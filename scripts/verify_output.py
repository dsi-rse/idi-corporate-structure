"""Verification script for corporate structure pipeline output.

Each check is independently selectable. Run all checks by default, or pass
--checks to run a specific subset. Use --list-checks to see what is available.

Usage:
    uv run python scripts/verify_output.py \
        --parquet output/latest.parquet \
        --failures failures/failures.json \
        --user-agent "Your Name your@email.com"

    # Run specific checks only:
    uv run python scripts/verify_output.py \
        --parquet output/latest.parquet \
        --failures failures/failures.json \
        --user-agent "Your Name your@email.com" \
        --checks structural location failures

    # List available checks:
    uv run python scripts/verify_output.py --list-checks
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from collections import Counter
from collections.abc import Callable

import pandas as pd
import requests
from idi_ftm2j_shared.api import SecClient

from idi_corporate_structure.extractor import _normalize, html_to_text
from idi_corporate_structure.pipeline import report_dates_by_accession

_DEFAULT_SAMPLE_SIZE = 30
_LOCATION_EMPTY_RATE_THRESHOLD = 0.15
_EXTRACTION_FAILED_RATE_THRESHOLD = 0.05
_GROUNDING_FAIL_THRESHOLD = 0.05
_LOCATION_GROUNDING_FAIL_THRESHOLD = 0.10
_LOCATION_WINDOW = 200

# Conservative lower bounds on subsidiary count for the most recent filing.
# Actual counts are higher; these catch silent extraction failures.
_KNOWN_CIKS: dict[str, tuple[str, int]] = {
    "320193": ("Apple Inc. (10-K, Exhibit 21)", 500),
    "789019": ("Microsoft Corporation (10-K, Exhibit 21)", 400),
    "97476": ("Toyota Motor Corporation (20-F, Exhibit 8)", 200),
}

_PASS = "[PASS]"  # noqa: S105
_FAIL = "[FAIL]"
_WARN = "[WARN]"
_INFO = "[INFO]"

_DEFAULT_RATE_LIMIT = 0.2
_DEFAULT_MAX_RETRIES = 3

log = logging.getLogger(__name__)


@dataclasses.dataclass
class VerifyContext:
    """Shared inputs passed to every check function."""

    df: pd.DataFrame
    failures_path: str
    sample_size: int
    sec_client: SecClient


@dataclasses.dataclass(frozen=True)
class Check:
    """A single named verification check."""

    name: str
    description: str
    fn: Callable[[VerifyContext], list[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    """Log a titled section separator."""
    log.info("\n%s", "=" * 60)
    log.info("  %s", title)
    log.info("%s", "=" * 60)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_structural(ctx: VerifyContext) -> list[str]:
    """Assert exhibit_url is non-null and exhibit_type matches form_type.

    Catches:
      - Rows where no exhibit URL was recorded (exhibit was never fetched).
      - Rows where the wrong exhibit number is assigned — 10-K must be exhibit
        21, 20-F must be exhibit 8.
    """
    _section("Structural integrity")
    failures: list[str] = []
    subs = ctx.df

    blank_url = subs["exhibit_url"].isna() | (subs["exhibit_url"].astype(str) == "")
    if blank_url.any():
        n = int(blank_url.sum())
        log.error("%s %d row(s) have a blank exhibit_url", _FAIL, n)
        log.error(
            "%s",
            subs[blank_url][["parent_cik", "accession_number", "name"]].to_string(index=False),
        )
        failures.append(f"{n} rows with blank exhibit_url")
    else:
        log.info("%s All %d rows have a non-empty exhibit_url", _PASS, len(subs))

    is_10k = subs["form_type"].str.match(r"10-?K", na=False)
    is_20f = subs["form_type"].str.match(r"20-?F", na=False)
    bad_10k = is_10k & (subs["exhibit_type"] != "21")
    bad_20f = is_20f & (subs["exhibit_type"] != "8")

    if bad_10k.any():
        n = int(bad_10k.sum())
        log.error("%s %d 10-K row(s) have exhibit_type != '21'", _FAIL, n)
        failures.append(f"{n} 10-K rows with wrong exhibit_type")
    else:
        log.info("%s All 10-K rows have exhibit_type='21'", _PASS)

    if bad_20f.any():
        n = int(bad_20f.sum())
        log.error("%s %d 20-F row(s) have exhibit_type != '8'", _FAIL, n)
        failures.append(f"{n} 20-F rows with wrong exhibit_type")
    else:
        log.info("%s All 20-F rows have exhibit_type='8'", _PASS)

    other = ~is_10k & ~is_20f
    if other.any():
        log.warning("%s %d row(s) have an unexpected form_type:", _WARN, int(other.sum()))
        log.warning("%s", subs[other]["form_type"].value_counts().to_string())

    return failures


@dataclasses.dataclass
class _GroundingCounts:
    """Accumulated pass/total counts across all exhibits in a grounding check run."""

    total_names: int = 0
    passed_names: int = 0
    total_locations: int = 0
    passed_locations: int = 0


def _fetch_exhibit_text(url: str, sec_client: SecClient) -> str:
    """Fetch an exhibit URL through ``SecClient`` and return its plain text.

    Spacing between requests is enforced via ``sec_client.rate_limit()`` and
    transient ``429``/``5xx`` responses are retried with exponential backoff
    (``Retry-After`` honored) by the underlying ``urllib3.Retry`` adapter.

    Raises:
        RuntimeError: when the client returns an ``error`` (network or HTTP).
    """
    result = sec_client.query_endpoint(url, return_json=False)
    if "error" in result:
        raise RuntimeError(result["error"])
    return html_to_text(result["data"])


def _name_in_plain(name_norm: str, plain_norm: str) -> bool:
    """Return True if the pre-normalized name appears in the pre-normalized document."""
    return name_norm in plain_norm


def _location_near_name(name_norm: str, location: str, plain_norm: str) -> bool | None:
    """Return True if the location appears within _LOCATION_WINDOW chars of the name.

    Returns None when the name position cannot be found so the caller can skip
    the location count rather than recording a false failure.
    """
    name_pos = plain_norm.find(name_norm)
    if name_pos == -1:
        return None
    window = plain_norm[
        max(0, name_pos - _LOCATION_WINDOW) : name_pos + len(name_norm) + _LOCATION_WINDOW
    ]
    return _normalize(location) in window


def _check_exhibit_rows(
    group_df: pd.DataFrame,
    plain_norm: str,
    url: str,
    counts: _GroundingCounts,
) -> None:
    """Check every row in one exhibit group, updating counts and logging per-row warnings."""
    for _, row in group_df.iterrows():
        name = str(row["name"])
        name_norm = _normalize(name)

        counts.total_names += 1
        if _name_in_plain(name_norm, plain_norm):
            counts.passed_names += 1
        else:
            log.warning("%s Name '%s' not found in %s", _WARN, name, url)

        location = str(row.get("location", "") or "")
        if not location:
            continue

        counts.total_locations += 1
        result = _location_near_name(name_norm, location, plain_norm)
        if result is None:
            counts.total_locations -= 1  # name absent — can't score location
        elif result:
            counts.passed_locations += 1
        else:
            log.warning("%s Location '%s' not near '%s' in %s", _WARN, location, name, url)


def _report_grounding_rate(
    label: str,
    passed: int,
    total: int,
    threshold: float,
    num_exhibits: int,
) -> str | None:
    """Log a grounding rate line and return a failure string when the threshold is exceeded.

    Returns None (no failure) when total is zero or the rate is within threshold.
    """
    if not total:
        log.info("%s No %s to check in this sample", _INFO, label)
        return None
    ungrounded = total - passed
    rate = ungrounded / total
    status = _FAIL if rate > threshold else _PASS
    log.info(
        "%s %d/%d %s found across %d exhibit(s) — %.1f%% ungrounded (threshold: %.0f%%)",
        status,
        passed,
        total,
        label,
        num_exhibits,
        rate * 100,
        threshold * 100,
    )
    if rate > threshold:
        return (
            f"Ungrounded {label} rate {rate:.1%} exceeds {threshold:.0%}"
            f" ({ungrounded}/{total} {label} across {num_exhibits} exhibit(s))"
        )
    return None


def check_grounding(ctx: VerifyContext) -> list[str]:
    """Fetch each unique exhibit URL once and verify names and locations appear in it.

    Name check:
      - Verifies every subsidiary name appears in the plain-text exhibit (using the
        same _normalize + html_to_text used by the extractor).

    Location check:
      - For rows with a non-empty location, verifies the location appears within
        _LOCATION_WINDOW characters of the name in the plain text — matching the
        windowed check performed during extraction.

    Groups rows by exhibit_url so each document is fetched exactly once.
    --sample-size limits the number of unique exhibits fetched, not the number of
    rows — all names within a fetched exhibit are always checked.
    """
    eligible = ctx.df[ctx.df["exhibit_url"].astype(str).str.startswith("http")]
    groups = eligible.groupby("exhibit_url")
    urls = list(groups.groups.keys())

    if ctx.sample_size:
        urls = urls[: ctx.sample_size]

    row_count = int(eligible["exhibit_url"].isin(urls).sum())
    _section(f"Name + location grounding — {len(urls)} exhibit(s), {row_count} rows")

    if not urls:
        log.warning("%s No rows with fetchable exhibit URLs — skipping", _WARN)
        return []

    counts = _GroundingCounts()
    fetch_errors: list[tuple[str, str]] = []

    for url in urls:
        try:
            plain_norm = _normalize(_fetch_exhibit_text(url, ctx.sec_client))
            _check_exhibit_rows(groups.get_group(url), plain_norm, url, counts)
        except (RuntimeError, requests.RequestException) as exc:
            fetch_errors.append((str(url), str(exc)))

    if fetch_errors:
        log.warning("%s %d exhibit(s) could not be fetched:", _WARN, len(fetch_errors))
        for url, err in fetch_errors[:5]:
            log.warning("    %s: %s", url, err)

    failures: list[str] = []
    for msg in [
        _report_grounding_rate(
            "names",
            counts.passed_names,
            counts.total_names,
            _GROUNDING_FAIL_THRESHOLD,
            len(urls),
        ),
        _report_grounding_rate(
            "locations",
            counts.passed_locations,
            counts.total_locations,
            _LOCATION_GROUNDING_FAIL_THRESHOLD,
            len(urls),
        ),
    ]:
        if msg:
            failures.append(msg)

    return failures


def check_location(ctx: VerifyContext) -> list[str]:
    """Report empty subsidiary location rate and parent_location distribution.

    Catches:
      - GPT systematically failing to extract jurisdiction strings from exhibits.
      - Exhibits where the location column is absent (expect a higher empty rate
        for those companies specifically).

    Two fields are reported separately:
      - location: GPT-extracted subsidiary jurisdiction (free text, may be blank
        legitimately for some exhibits).
      - parent_location: raw SEC stateOfIncorporation code (not GPT-extracted);
        logged for manual inspection of domestic/foreign composition.
    """
    _section("Location quality")
    failures: list[str] = []
    subs = ctx.df

    empty_loc = subs["location"].isna() | (subs["location"].astype(str) == "")
    rate = float(empty_loc.mean())
    status = _FAIL if rate > _LOCATION_EMPTY_RATE_THRESHOLD else _PASS
    log.info(
        "%s Subsidiary location empty rate: %.1f%% (threshold: %.0f%%)",
        status,
        rate * 100,
        _LOCATION_EMPTY_RATE_THRESHOLD * 100,
    )
    if rate > _LOCATION_EMPTY_RATE_THRESHOLD:
        failures.append(
            f"Subsidiary location empty rate {rate:.1%} exceeds "
            f"{_LOCATION_EMPTY_RATE_THRESHOLD:.0%}"
        )

    # Per-exhibit breakdown — show exhibits with the highest empty location count
    if empty_loc.any():
        exhibit_stats = (
            subs.groupby("exhibit_url")
            .agg(
                total=("location", "count"),
                empty=("location", lambda s: (s.isna() | (s.astype(str) == "")).sum()),
            )
            .assign(empty_rate=lambda d: d["empty"] / d["total"])
            .query("empty > 0")
            .sort_values("empty", ascending=False)
            .head(10)
        )
        log.info("\n%s Top exhibits by empty location count (up to 10):", _INFO)
        for url, row in exhibit_stats.iterrows():
            log.info(
                "  %3d / %3d empty (%.0f%%)  %s",
                int(row["empty"]),
                int(row["total"]),
                row["empty_rate"] * 100,
                url,
            )

    log.info("\n%s parent_location top 20 (raw SEC stateOfIncorporation codes):", _INFO)
    log.info("%s", subs["parent_location"].value_counts().head(20).to_string())

    return failures


def check_failures(ctx: VerifyContext) -> list[str]:
    """Load failures.json and report the failure type distribution.

    Catches:
      - Elevated extraction_failed rates indicating GPT is not returning
        structured data (prompt issue, model change, or document size).
      - Unexpected failure patterns after a code or configuration change.

    failures.json schema (written by FailureRegistry.save):
      {"entries": [[cik, filename], ...], "reasons": {"cik filename": "type"}}
    """
    _section("Failure distribution")
    issues: list[str] = []

    try:
        with pathlib.Path(ctx.failures_path).open() as f:
            data = json.load(f)
    except FileNotFoundError:
        log.warning("%s failures.json not found at %s — skipping", _WARN, ctx.failures_path)
        return issues
    except json.JSONDecodeError as exc:
        log.error("%s Could not parse failures.json: %s", _FAIL, exc)
        return [f"failures.json parse error: {exc}"]

    reasons = data.get("reasons", {})
    total = len(reasons)
    log.info("%s Total permanent failures recorded: %d", _INFO, total)

    if not reasons:
        log.info("%s No failures recorded", _PASS)
        return issues

    counts = Counter(reasons.values())
    log.info("\n%s Failure type breakdown:", _INFO)
    for failure_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        log.info("  %-40s %8d  (%.1f%%)", failure_type, count, pct)

    extraction_failed = counts.get("extraction_failed", 0)
    rate = extraction_failed / total
    status = _FAIL if rate > _EXTRACTION_FAILED_RATE_THRESHOLD else _PASS
    log.info(
        "\n%s extraction_failed rate: %.1f%% (threshold: %.0f%%)",
        status,
        rate * 100,
        _EXTRACTION_FAILED_RATE_THRESHOLD * 100,
    )
    if rate > _EXTRACTION_FAILED_RATE_THRESHOLD:
        issues.append(f"High extraction_failed rate: {rate:.1%} of all failures")

    return issues


def _fetch_report_dates(cik: str, sec_client: SecClient) -> dict[str, str]:
    """Return the accession -> reportDate map for one CIK, or {} if unavailable."""
    url = f"https://data.sec.gov/submissions/CIK{str(int(cik)).zfill(10)}.json"
    result = sec_client.query_endpoint(sec_url=url)
    if "error" in result:
        log.warning("%s Could not fetch submissions JSON for CIK %s", _WARN, cik)
        return {}
    return report_dates_by_accession(result.get("data", {}))


def check_period(ctx: VerifyContext) -> list[str]:
    """Verify period_of_report is present and agrees with EDGAR.

    Catches:
      - Rows whose reporting period disagrees with EDGAR's reportDate for the
        same accession — the failure mode that would silently misdate a
        subsidiary list.
      - Blank periods, reported as a warning rather than a failure: rows written
        before the column existed are legitimately blank until they are
        backfilled offline.
    """
    _section("Period of report")
    failures: list[str] = []
    subs = ctx.df

    if "period_of_report" not in subs.columns:
        log.error("%s Output has no period_of_report column", _FAIL)
        return ["missing period_of_report column"]

    blank = subs["period_of_report"].isna() | (subs["period_of_report"].astype(str) == "")
    if blank.any():
        n = int(blank.sum())
        log.warning(
            "%s %d of %d row(s) have a blank period_of_report (expected for rows written "
            "before the column existed; clears once they are backfilled)",
            _WARN,
            n,
            len(subs),
        )
    else:
        log.info("%s All %d rows have a non-empty period_of_report", _PASS, len(subs))

    # Spot-check the CIKs that motivated the column: a delinquent filer and a
    # co-registrant filing whose accession lives under another CIK's folder.
    for cik in ("1583994", "857501"):
        rows = subs[subs["parent_cik"].astype(str).str.lstrip("0") == cik]
        if rows.empty:
            log.info("%s CIK %s not present in output — skipping spot check", _INFO, cik)
            continue

        expected = _fetch_report_dates(cik, ctx.sec_client)
        if not expected:
            continue

        for accession, group in rows.groupby("accession_number"):
            actual = str(group["period_of_report"].iloc[0])
            want = expected.get(str(accession))
            if want is None:
                log.warning("%s EDGAR has no reportDate for %s", _WARN, accession)
            elif not actual:
                log.warning("%s %s has a blank period (EDGAR: %s)", _WARN, accession, want)
            elif actual != want:
                log.error(
                    "%s %s has period_of_report=%s but EDGAR reports %s",
                    _FAIL,
                    accession,
                    actual,
                    want,
                )
                failures.append(f"{accession}: period {actual} != EDGAR {want}")
            else:
                log.info("%s %s period_of_report=%s matches EDGAR", _PASS, accession, actual)

    return failures


# ---------------------------------------------------------------------------
# Check registry — controls ordering and --list-checks output
# ---------------------------------------------------------------------------

CHECKS: list[Check] = [
    Check(
        name="structural",
        description="Assert exhibit_url is non-null and exhibit_type matches form_type",
        fn=check_structural,
    ),
    Check(
        name="grounding",
        description="Fetch each unique exhibit URL once and verify all subsidiary names appear in it",
        fn=check_grounding,
    ),
    Check(
        name="location",
        description="Report empty subsidiary location rate and parent_location distribution",
        fn=check_location,
    ),
    Check(
        name="period",
        description="Verify period_of_report is present and matches EDGAR's reportDate",
        fn=check_period,
    ),
    Check(
        name="failures",
        description="Load failures.json and report failure type distribution",
        fn=check_failures,
    ),
]

_CHECK_NAMES: list[str] = [c.name for c in CHECKS]


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------


def run_checks(ctx: VerifyContext, checks: list[Check]) -> list[str]:
    """Run each check in sequence and return all collected failure messages.

    Args:
        ctx: Shared pipeline output and configuration for all checks.
        checks: Ordered list of checks to execute.

    Returns:
        Flat list of failure message strings across all checks.
    """
    all_failures: list[str] = []
    for check in checks:
        all_failures += check.fn(ctx)
    return all_failures


def print_summary(all_failures: list[str]) -> None:
    """Print the summary of the checks."""
    _section("Summary")
    if all_failures:
        log.error("%s %d check(s) failed:\n", _FAIL, len(all_failures))
        for item in all_failures:
            log.error("  - %s", item)
        sys.exit(1)
    else:
        log.info("%s All checks passed", _PASS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def create_args() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Verify SubsidiaryPipeline output (Parquet + failures.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parquet", help="Path to latest.parquet")
    parser.add_argument("--failures", help="Path to failures.json")
    parser.add_argument(
        "--user-agent",
        metavar="STRING",
        help="User-Agent header sent with SEC EDGAR requests (required by EDGAR policy)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=(
            f"Max unique exhibits to fetch for the grounding check (default: {_DEFAULT_SAMPLE_SIZE}). "
            "All names within each fetched exhibit are checked. Set to 0 to fetch every exhibit."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=_DEFAULT_RATE_LIMIT,
        metavar="SECONDS",
        help=(
            f"Minimum seconds between SEC requests (default: {_DEFAULT_RATE_LIMIT}). "
            "Increase if SEC EDGAR throttles a full-corpus run."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=_DEFAULT_MAX_RETRIES,
        help=(
            f"Max retries for transient 429/5xx responses (default: {_DEFAULT_MAX_RETRIES}). "
            "Backoff is exponential and Retry-After is honored."
        ),
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=_CHECK_NAMES,
        metavar="CHECK",
        default=_CHECK_NAMES,
        help=(
            f"One or more checks to run. Choices: {', '.join(_CHECK_NAMES)}. "
            "Default: all. Use --list-checks to see descriptions."
        ),
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="Print available checks with descriptions and exit",
    )
    return parser


def main() -> None:
    """Parse arguments, load data, run selected checks, and exit with status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = create_args()
    args = parser.parse_args()

    if args.list_checks:
        print("\nAvailable checks:\n")
        for check in CHECKS:
            print(f"  {check.name:<15}  {check.description}")
        print()
        sys.exit(0)

    missing = [
        flag
        for flag, value in (
            ("--parquet", args.parquet),
            ("--failures", args.failures),
            ("--user-agent", args.user_agent),
        )
        if not value
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    # Load the results parquet file
    log.info("\nLoading %s ...", args.parquet)
    try:
        subs = pd.read_parquet(args.parquet)
    except FileNotFoundError:
        log.error("%s Parquet file not found: %s", _FAIL, args.parquet)
        sys.exit(1)

    log.info(
        "%s %d rows across %d unique parent CIKs", _INFO, len(subs), subs["parent_cik"].nunique()
    )

    # Build a SEC client with retry/backoff and a per-instance User-Agent
    sec_client = SecClient(rate_limit=args.rate_limit, user_agent=args.user_agent)
    sec_client.max_retries = args.max_retries

    # Create an object to hold the context for the checks
    ctx = VerifyContext(
        df=subs,
        failures_path=args.failures,
        sample_size=args.sample_size,
        sec_client=sec_client,
    )

    # Run selected checks
    selected = [c for c in CHECKS if c.name in set(args.checks)]
    all_failures = run_checks(ctx, selected)

    # Print the summary
    print_summary(all_failures)


if __name__ == "__main__":
    main()
