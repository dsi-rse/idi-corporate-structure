"""Pipeline Orchestrator - Runs the corporate structure pipeline for the specified input file.

Uses the `SubsidiaryPipeline` to process the input file and save the results to the output file.
GPT extraction is performed in a separate thread.

The orchestrator is responsible for running the pipeline.
"""

# Standard imports
import argparse
import datetime
import os
import sys

# Third party imports
import pandas as pd
from idi_ftm2j_shared.api import SecClient
from idi_ftm2j_shared.logs import get_logger

# Application imports
from idi_corporate_structure.extractor import GptExtractor
from idi_corporate_structure.pipeline import SubsidiaryPipeline
from idi_corporate_structure.types import PipelineConfig

DEFAULT_LOOK_BACK = 7
SENSITIVE_ARGS = {"openai_api_key", "sec_user_agent"}


def valid_date(s: str) -> datetime.date:
    """Parse a ``YYYY-MM-DD`` string into a date, for use as an argparse ``type``.

    Args:
        s: Date string to parse.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If ``s`` is not a valid ISO date.
    """
    try:
        return datetime.date.fromisoformat(s)  # expects YYYY-MM-DD
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Not a valid date: {s!r}") from err


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Enforce the pairing argparse can't express on its own.

    Args:
        args: Parsed command-line arguments to validate, mutated in place to
            apply the ``--look-back`` default in daily mode.
        parser: Parser used to report validation errors via ``parser.error``,
            which prints usage and exits.

    Returns:
        None
    """
    if args.start_date and not args.end_date:
        parser.error("--end-date is required when --start-date is given")
    if args.daily and args.end_date:
        parser.error("--end-date cannot be used with --daily")
    if args.start_date and args.end_date and args.end_date < args.start_date:
        parser.error("--end-date must not be before --start-date")

    # look-back only makes sense in daily mode
    if args.look_back is not None and not args.daily:
        parser.error("--look-back can only be used with --daily")
    if args.look_back is not None and (args.start_date or args.end_date):
        parser.error("--look-back cannot be used with --start-date/--end-date")

    if not args.openai_api_key:
        parser.error("OpenAI API key required via --openai-api-key or OPENAI_API_KEY env var")
    if not args.sec_user_agent:
        parser.error("SEC User-Agent required via --sec-user-agent or SEC_USER_AGENT env var")

    # apply default AFTER the conflict checks, only in daily mode
    if args.daily and args.look_back is None:
        args.look_back = DEFAULT_LOOK_BACK


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Corporate Structure Pipeline Orchestrator")
    parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    parser.add_argument("--failure-file", type=str, required=True, help="Failure file path")
    parser.add_argument(
        "--sec-bucket-prefix",
        type=str,
        required=True,
        help="S3 Bucket and prefix that contains SEC data (bucket-name/prefix)",
    )

    # daily flag cannot be combined with start date
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daily", action="store_true", help="Scrape most recent filing")

    mode.add_argument("--start-date", type=valid_date, help="Range start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=valid_date, help="Range end (YYYY-MM-DD)")

    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", ""),
        help="OpenAI model ID for extraction (falls back to OPENAI_MODEL env var, then default)",
    )

    parser.add_argument(
        "--sec-user-agent",
        type=str,
        default=os.environ.get("SEC_USER_AGENT", ""),
        help="SEC EDGAR User-Agent header value (falls back to SEC_USER_AGENT env var)",
    )

    parser.add_argument("--rate-limit", type=float, default=0.2, help="Rate limit")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Flush extracted results to the output file every N documents (0 disables)",
    )
    parser.add_argument(
        "--look-back",
        type=int,
        default=None,
        help="Days to look back from the most recent date (daily mode only)",
    )

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def get_dates(args: argparse.Namespace) -> tuple[datetime.date, datetime.date]:
    """Resolve the start/end date range to scrape from the parsed arguments.

    In ``--daily`` mode, reads the most recent ``date_scraped`` from the SEC
    bucket's manifest and looks back ``args.look_back`` days from there. The
    window is derived from the scraped date so it stays consistent with the
    ``search_by="scraped_date"`` filter the pipeline uses. Otherwise returns the
    explicit ``--start-date``/``--end-date`` values.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of ``(start_date, end_date)``.

    Raises:
        ValueError: If ``--daily`` mode is used and the manifest has no
            usable ``date_scraped`` values.
    """
    if not args.daily:
        return args.start_date, args.end_date

    manifest_df = pd.read_parquet(f"s3://{args.sec_bucket_prefix}/manifest.parquet")
    latest = manifest_df["date_scraped"].max()
    if pd.isna(latest):
        raise ValueError("manifest.parquet has no usable date_scraped values")
    end_date = pd.to_datetime(latest).date()
    start_date = end_date - datetime.timedelta(days=args.look_back)
    return start_date, end_date


def main() -> None:
    """Main function to run the pipeline orchestrator."""
    start = datetime.datetime.now()
    args = get_args()

    logger = get_logger("orchestrator")
    for key, value in vars(args).items():
        shown = "********************" if key in SENSITIVE_ARGS and value else value
        logger.info("%s = %r", key, shown)

    start_date, end_date = get_dates(args)
    if pd.isna(start_date) or pd.isna(end_date):
        logger.error("Could not locate start and end dates from command line arguments.")
        sys.exit(1)

    logger.info("Searching for date range: %s - %s", start_date, end_date)
    config = PipelineConfig(
        output_file=args.output_file,
        failure_file=args.failure_file,
        start_date=start_date,
        end_date=end_date,
        sec_bucket=args.sec_bucket_prefix.split("/")[0],
        rate_limit=args.rate_limit,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        openai_api_key=args.openai_api_key,
    )
    sec_client = SecClient(rate_limit=config.rate_limit, user_agent=args.sec_user_agent)
    extractor = GptExtractor(openai_api_key=config.openai_api_key, model=args.model)
    pipeline = SubsidiaryPipeline(config=config, sec_client=sec_client, extractor=extractor)
    pipeline.run()

    end = datetime.datetime.now()
    print(f"Elasped time: {end - start}")


if __name__ == "__main__":
    main()
