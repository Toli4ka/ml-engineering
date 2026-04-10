from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, TypeVar
import re
import logging


import pandas as pd
from mcp.server.fastmcp import FastMCP

# FastMCP auto-generates tool schemas from type hints/docstrings.
mcp = FastMCP("Timesheet Validator", json_response=True)

REQUIRED_COLUMNS = [
    "employee_id",
    "employee_name",
    "date",
    "project_id",
    "hours_logged",
]

TEXT_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty one": 21,
    "twenty-one": 21,
    "twenty two": 22,
    "twenty-two": 22,
    "twenty three": 23,
    "twenty-three": 23,
    "twenty four": 24,
    "twenty-four": 24,
}

PROJECT_ID_PATTERN = re.compile(r"PRJ-\d{3}")
MAX_CACHE_ENTRIES = 8


LOG_PATH = Path(__file__).with_name("logs") / "server.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size: int
    mtime_ns: int


@dataclass
class AnalysisResult:
    fingerprint: FileFingerprint
    row_count: int
    missing_columns: list[str]
    issues: list[dict[str, Any]]
    error: str | None = None


@dataclass
class SuggestionResult:
    fingerprint: FileFingerprint
    issue_count: int
    suggestions: list[dict[str, Any]]



T = TypeVar("T", AnalysisResult, SuggestionResult)
_analysis_cache: OrderedDict[str, AnalysisResult] = OrderedDict()
_suggestion_cache: OrderedDict[str, SuggestionResult] = OrderedDict()


@lru_cache(maxsize=1)
def load_valid_project_ids() -> set[str]:
    return {f"PRJ-{number:03d}" for number in range(100)}


def _resolve_csv_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _default_fixed_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.fixed{csv_path.suffix}")


def _build_fingerprint(csv_path: Path) -> FileFingerprint:
    stats = csv_path.stat()
    return FileFingerprint(
        path=str(csv_path),
        size=stats.st_size,
        mtime_ns=stats.st_mtime_ns,
    )

def _cache_get(
    cache: OrderedDict[str, T],
    path: str,
    fingerprint: FileFingerprint,
) -> T | None:
    cached = cache.get(path)
    if cached is None:
        return None
    if cached.fingerprint != fingerprint:
        cache.pop(path, None)
        return None
    cache.move_to_end(path)
    return cached

def _cache_put(cache: OrderedDict[str, T], path: str, value: T) -> T:
    cache[path] = value
    cache.move_to_end(path)
    while len(cache) > MAX_CACHE_ENTRIES:
        cache.popitem(last=False)
    return value


def read_csv_file(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {csv_path}") from exc


def add_row_numbers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_row_number"] = range(2, len(result) + 2)
    return result


def get_missing_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def run_timesheet_validation(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_mask = parsed_dates.isna()
    for _, row in df.loc[invalid_date_mask].iterrows():
        issues.append(
            {
                "row": int(row["_row_number"]),
                "column": "date",
                "issue_type": "invalid_date",
                "value": row["date"],
                "severity": "error",
                "message": f"Invalid date: {row['date']!r}",
            }
        )

    parsed_hours = pd.to_numeric(df["hours_logged"], errors="coerce")
    non_numeric_hours_mask = parsed_hours.isna()
    for _, row in df.loc[non_numeric_hours_mask].iterrows():
        issues.append(
            {
                "row": int(row["_row_number"]),
                "column": "hours_logged",
                "issue_type": "non_numeric_hours",
                "value": row["hours_logged"],
                "severity": "error",
                "message": f"Non-numeric hours_logged: {row['hours_logged']!r}",
            }
        )

    valid_hours_mask = ~parsed_hours.isna()
    out_of_range_mask = valid_hours_mask & ((parsed_hours < 0) | (parsed_hours > 24))
    for idx, row in df.loc[out_of_range_mask].iterrows():
        issues.append(
            {
                "row": int(row["_row_number"]),
                "column": "hours_logged",
                "issue_type": "hours_out_of_range",
                "value": row["hours_logged"],
                "severity": "error",
                "message": f"hours_logged must be between 0 and 24, got {parsed_hours.loc[idx]}",
            }
        )

    duplicate_cols = ["employee_id", "date", "project_id", "hours_logged"]
    duplicate_mask = df.duplicated(subset=duplicate_cols, keep=False)
    for _, row in df.loc[duplicate_mask].iterrows():
        issues.append(
            {
                "row": int(row["_row_number"]),
                "column": ",".join(duplicate_cols),
                "issue_type": "duplicate_row",
                "value": row["_row_number"],
                "severity": "warning",
                "message": "Potential duplicate timesheet row.",
            }
        )

    invalid_project_id_mask = ~df["project_id"].astype(str).str.fullmatch(PROJECT_ID_PATTERN)
    for _, row in df.loc[invalid_project_id_mask].iterrows():
        issues.append(
            {
                "row": int(row["_row_number"]),
                "column": "project_id",
                "issue_type": "invalid_project_id_format",
                "value": row["project_id"],
                "severity": "error",
                "message": f"Invalid project_id format: {row['project_id']!r}. Expected PRJ-XXX where XXX is a number.",
            }
        )

    return issues

def analyze_timesheet(path: str) -> AnalysisResult:
    csv_path = _resolve_csv_path(path)
    cache_key = str(csv_path)

    if not csv_path.exists():
        return AnalysisResult(
            fingerprint=FileFingerprint(path=cache_key, size=-1, mtime_ns=-1),
            row_count=0,
            missing_columns=[],
            issues=[],
            error=f"File not found: {csv_path}",
        )

    fingerprint = _build_fingerprint(csv_path)
    cached = _cache_get(_analysis_cache, cache_key, fingerprint)
    if cached is not None:
        return cached

    try:
        df = read_csv_file(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        analysis = AnalysisResult(
            fingerprint=fingerprint,
            row_count=0,
            missing_columns=[],
            issues=[],
            error=str(exc),
        )
        return _cache_put(_analysis_cache, cache_key, analysis)

    missing_columns = get_missing_columns(df)
    if missing_columns:
        analysis = AnalysisResult(
            fingerprint=fingerprint,
            row_count=int(len(df)),
            missing_columns=missing_columns,
            issues=[],
        )
        return _cache_put(_analysis_cache, cache_key, analysis)

    prepared_df = add_row_numbers(df)
    analysis = AnalysisResult(
        fingerprint=fingerprint,
        row_count=int(len(prepared_df)),
        missing_columns=[],
        issues=run_timesheet_validation(prepared_df),
    )
    return _cache_put(_analysis_cache, cache_key, analysis)


def build_validation_response(analysis: AnalysisResult) -> dict[str, Any]:
    if analysis.error:
        return {"status": "error", "message": analysis.error, "issues": []}

    if analysis.missing_columns:
        return {
            "status": "error",
            "message": "Missing required columns.",
            "missing_columns": analysis.missing_columns,
            "issues": [],
        }

    return {
        "status": "ok",
        "message": "Validation completed.",
        "file": analysis.fingerprint.path,
        "row_count": analysis.row_count,
        "issue_count": int(len(analysis.issues)),
        "error_count": int(sum(1 for x in analysis.issues if x["severity"] == "error")),
        "warning_count": int(sum(1 for x in analysis.issues if x["severity"] == "warning")),
        "issues": analysis.issues[:100], #TODO: is it a good idea to take just first 100 issues for production code? 
    }



def _make_suggestion(
    issue: dict[str, Any],
    original_value: Any,
    suggested_value: Any,
    action: str,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "row": issue["row"],
        "column": issue["column"],
        "issue_type": issue["issue_type"],
        "original_value": original_value,
        "suggested_value": suggested_value,
        "action": action,
        "confidence": confidence,
        "reason": reason,
    }


def _suggest_for_invalid_date(issue: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    raw = str(value).strip()

    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", raw):
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=raw.replace("/", "-"),
            action="auto_fix",
            confidence=0.99,
            reason="Normalized date separator from '/' to '-' in ISO-style date.",
        )

    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", raw)
    if m:
        dd, mm, yyyy = m.groups()
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=f"{yyyy}-{mm}-{dd}",
            action="auto_fix",
            confidence=0.96,
            reason="Converted date from DD.MM.YYYY to YYYY-MM-DD.",
        )

    if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", raw):
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=raw.replace(".", "-"),
            action="auto_fix",
            confidence=0.97,
            reason="Normalized date separator from '.' to '-' in ISO-style date.",
        )

    return None


def _suggest_for_non_numeric_hours(issue: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    raw = str(value).strip().lower()

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*h", raw)
    if m:
        number = float(m.group(1))
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=int(number) if number.is_integer() else number,
            action="auto_fix",
            confidence=0.99,
            reason="Removed trailing hour unit and parsed numeric value.",
        )

    if raw in TEXT_NUMBERS:
        number = TEXT_NUMBERS[raw]
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=number,
            action="suggest_only",
            confidence=0.85,
            reason="Converted written number to numeric value.",
        )

    if re.fullmatch(r"\d+,\d+", raw):
        number = float(raw.replace(",", "."))
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=number,
            action="auto_fix",
            confidence=0.95,
            reason="Converted decimal comma to decimal point and parsed numeric value.",
        )

    return None


def _suggest_for_hours_out_of_range(issue: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    raw = str(value).strip().lower()

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*h", raw)
    if m:
        number = float(m.group(1))
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=int(number) if number.is_integer() else number,
            action="suggest_only",
            confidence=0.70,
            reason="Parsed numeric value, but hours remain outside allowed range and need review.",
        )

    return None


def _suggest_for_invalid_project_id_format(
    issue: dict[str, Any],
    value: Any,
    valid_project_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None

    raw = str(value).strip().upper().replace(" ", "")

    if re.fullmatch(r"[A-Z]{3}\d{3,}", raw):
        normalized = f"{raw[:3]}-{raw[3:]}"
        if valid_project_ids and normalized not in valid_project_ids:
            return _make_suggestion(
                issue,
                original_value=value,
                suggested_value=normalized,
                action="suggest_only",
                confidence=0.75,
                reason="Normalized project ID format, but value was not found in known project registry.",
            )
        return _make_suggestion(
            issue,
            original_value=value,
            suggested_value=normalized,
            action="auto_fix",
            confidence=0.97,
            reason="Inserted expected separator into project ID format.",
        )

    return None


def _suggest_for_duplicate_row(issue: dict[str, Any], value: Any) -> dict[str, Any] | None:
    return _make_suggestion(
        issue,
        original_value=value,
        suggested_value=None,
        action="suggest_only",
        confidence=0.60,
        reason="Potential duplicate row detected; review before removing because duplicates may be legitimate.",
    )


def build_suggestions(analysis: AnalysisResult) -> SuggestionResult:
    cached = _cache_get(_suggestion_cache, analysis.fingerprint.path, analysis.fingerprint)
    if cached is not None:
        return cached

    suggestions: list[dict[str, Any]] = []
    valid_project_ids = load_valid_project_ids()

    for issue in analysis.issues:
        issue_type = issue["issue_type"]
        value = issue["value"]
        suggestion: dict[str, Any] | None = None

        if issue_type == "invalid_date":
            suggestion = _suggest_for_invalid_date(issue, value)
        elif issue_type == "non_numeric_hours":
            suggestion = _suggest_for_non_numeric_hours(issue, value)
        elif issue_type == "hours_out_of_range":
            suggestion = _suggest_for_hours_out_of_range(issue, value)
        elif issue_type == "invalid_project_id_format":
            suggestion = _suggest_for_invalid_project_id_format(issue, value, valid_project_ids)
        elif issue_type == "duplicate_row":
            suggestion = _suggest_for_duplicate_row(issue, value)

        if suggestion is not None:
            suggestions.append(suggestion)

    result = SuggestionResult(
        fingerprint=analysis.fingerprint,
        issue_count=len(analysis.issues),
        suggestions=suggestions,
    )
    return _cache_put(_suggestion_cache, analysis.fingerprint.path, result)


def build_suggestion_response(
    analysis: AnalysisResult,
    suggestion_result: SuggestionResult | None = None,
) -> dict[str, Any]:
    if analysis.error:
        return {"status": "error", "message": analysis.error, "issues": []}

    if analysis.missing_columns:
        return {
            "status": "error",
            "message": "Missing required columns.",
            "missing_columns": analysis.missing_columns,
            "issues": [],
        }

    suggestion_result = suggestion_result or SuggestionResult(
        fingerprint=analysis.fingerprint,
        issue_count=len(analysis.issues),
        suggestions=[],
    )

    return {
        "status": "ok",
        "message": "Fix suggestions generated.",
        "file": analysis.fingerprint.path,
        "issue_count": suggestion_result.issue_count,
        "suggestion_count": len(suggestion_result.suggestions),
        "auto_fix_count": sum(1 for s in suggestion_result.suggestions if s["action"] == "auto_fix"),
        "suggest_only_count": sum(1 for s in suggestion_result.suggestions if s["action"] == "suggest_only"),
        "suggestions": suggestion_result.suggestions[:100],
    }


def _apply_suggestions_to_dataframe(
    df: pd.DataFrame,
    suggestion_result: SuggestionResult,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_df = df.copy()
    applied_fixes: list[dict[str, Any]] = []
    skipped_fixes: list[dict[str, Any]] = []

    for suggestion in suggestion_result.suggestions:
        if suggestion["action"] != "auto_fix":
            skipped_fixes.append(suggestion)
            continue

        row_number = int(suggestion["row"])
        row_index = row_number - 2
        column = suggestion["column"]

        if row_index < 0 or row_index >= len(fixed_df) or column not in fixed_df.columns:
            skipped_fixes.append(
                {
                    **suggestion,
                    "skip_reason": "Could not map suggestion back to a writable CSV cell.",
                }
            )
            continue

        if str(fixed_df[column].dtype) != "object": #TODO: Explain what this check does? 
            fixed_df[column] = fixed_df[column].astype(object)
        fixed_df.at[row_index, column] = suggestion["suggested_value"]
        applied_fixes.append(suggestion)

    return fixed_df, applied_fixes, skipped_fixes


def _write_fixed_timesheet(
    source_path: Path,
    fixed_df: pd.DataFrame,
) -> Path:
    destination = _default_fixed_output_path(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixed_df.to_csv(destination, index=False)
    return destination


@mcp.tool()
def validate_timesheet_csv(path: str) -> dict[str, Any]:
    """
    Validate a timesheet CSV and return a structured report.

    Expected columns:
    - employee_id
    - employee_name
    - date
    - project_id
    - hours_logged

    Rules:
    - required columns must exist
    - date must be parseable
    - hours_logged must be numeric
    - hours_logged must be between 0 and 24 inclusive
    - duplicate rows are flagged
    """
    analysis = analyze_timesheet(path)
    return build_validation_response(analysis)


@mcp.tool()
def suggest_safe_fixes(path: str) -> dict[str, Any]:
    """
    Suggest safe fixes for a timesheet CSV based on validation issues.

    Uses shared analysis artifacts and reuses cached suggestions when the
    target file has not changed. This tool does not modify the file.
    """
    analysis = analyze_timesheet(path)
    if analysis.error or analysis.missing_columns:
        return build_suggestion_response(analysis)

    suggestion_result = build_suggestions(analysis)
    return build_suggestion_response(analysis, suggestion_result)


@mcp.tool()
def fix_timesheet(path: str) -> dict[str, Any]:
    """
    Apply safe automatic fixes to a timesheet CSV.

    This writes a new sibling file with a .fixed suffix and leaves the source
    file unchanged.
    """
    analysis = analyze_timesheet(path)
    if analysis.error:
        return {"status": "error", "message": analysis.error, "applied_fixes": []}

    if analysis.missing_columns:
        return {
            "status": "error",
            "message": "Missing required columns.",
            "missing_columns": analysis.missing_columns,
            "applied_fixes": [],
        }

    source_path = _resolve_csv_path(path)
    source_df = read_csv_file(source_path)
    suggestion_result = build_suggestions(analysis)
    fixed_df, applied_fixes, skipped_fixes = _apply_suggestions_to_dataframe(
        source_df,
        suggestion_result,
    )

    if not applied_fixes:
        return {
            "status": "ok",
            "message": "No automatic fixes were applied.",
            "file": analysis.fingerprint.path,
            "issue_count": len(analysis.issues),
            "suggestion_count": len(suggestion_result.suggestions),
            "applied_fix_count": 0,
            "skipped_fix_count": len(skipped_fixes),
            "applied_fixes": [],
            "skipped_fixes": skipped_fixes[:2], #TODO: Do I need to return skipped fixes? It eats a lot of context 
        }

    destination = _write_fixed_timesheet(source_path, fixed_df)

    post_fix_analysis = analyze_timesheet(str(destination))

    return {
        "status": "ok",
        "message": "Safe fixes applied.",
        "source_file": analysis.fingerprint.path,
        "output_file": str(destination),
        "issue_count_before": len(analysis.issues),
        "issue_count_after": len(post_fix_analysis.issues),
        "applied_fix_count": len(applied_fixes),
        "skipped_fix_count": len(skipped_fixes),
        "applied_fixes": applied_fixes[:100],
        "skipped_fixes": skipped_fixes[:100],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
