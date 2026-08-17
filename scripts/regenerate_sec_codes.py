r"""Regenerate the SEC_STATE_OF_INCORPORATION dict in normalization.py.

Fetches the SEC EDGAR state/country code page, parses the HTML tables, and
prints a Python dict literal to stdout. The output is meant to be pasted into
``src/idi_corporate_structure/processor/normalization.py``.

Run by hand when the SEC table changes (rare — the underlying ISO 3166 list
updates infrequently and additions don't break existing rows because
``normalize_parent_location`` passes through unknown codes).

Usage::

    uv run python scripts/regenerate_sec_codes.py \
        --user-agent "Your Name email@example.com"
"""

# Standard application imports
import argparse
import html
import re
import sys
import urllib.request

SEC_URL = "https://www.sec.gov/submit-filings/filer-support-resources/edgar-state-country-codes"
_PERMITTED_SCHEMES = ("https://", "http://")
_MIN_EXPECTED_CODES = 250
_MAX_CODE_LENGTH = 3
_EXPECTED_TABLE_CELLS = 2

_FIXES = {
    " And ": " and ",
    " Of ": " of ",
    " The ": " the ",
    "'S ": "'s ",
}


def _titlecase(name: str) -> str:
    out = name.title()
    for k, v in _FIXES.items():
        out = out.replace(k, v)
    if out.endswith(" The"):
        out = out[: -len(" The")] + " the"
    if out.endswith("'S"):
        out = out[:-2] + "'s"
    return out


def fetch(user_agent: str) -> str:
    """Fetch the SEC EDGAR state/country code page and return its HTML."""
    if not any(SEC_URL.startswith(scheme) for scheme in _PERMITTED_SCHEMES):
        raise ValueError(f"Unexpected URL scheme: {SEC_URL}")
    req = urllib.request.Request(SEC_URL, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def parse(page: str) -> dict[str, str]:
    """Parse the SEC code HTML page and return a {code: name} mapping."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.DOTALL)
    pairs: dict[str, str] = {}
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)
        cleaned = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
        if len(cleaned) != _EXPECTED_TABLE_CELLS:
            continue
        code, name = cleaned
        if not code or len(code) > _MAX_CODE_LENGTH or not name:
            continue
        if name.upper() in {"STATE", "COUNTRY", "CANADIAN PROVINCE"}:
            continue
        pairs[code] = _titlecase(name)
    return pairs


def main() -> int:
    """Entry point: fetch, parse, and print the SEC code dict to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-agent",
        required=True,
        help="User-Agent header for the SEC fetch (e.g. 'Name email@example.com').",
    )
    args = parser.parse_args()

    page = fetch(args.user_agent)
    codes = parse(page)
    if len(codes) < _MIN_EXPECTED_CODES:
        print(f"WARNING: only {len(codes)} codes parsed — expected ~310", file=sys.stderr)

    print("SEC_STATE_OF_INCORPORATION: dict[str, str] = {")
    for code, name in codes.items():
        escaped = name.replace('"', '\\"')
        print(f'    "{code}": "{escaped}",')
    print("}")
    print(f"# Total: {len(codes)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
