from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class IssueSuggestion:
    issue_type: str
    raw_value: str | None
    suggested_value: str | None
    suggestion_reason: str
    confidence: Decimal

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["confidence"] = str(self.confidence)
        return row


NUMBER_WORDS = {
    "zero": Decimal("0"),
    "one": Decimal("1"),
    "two": Decimal("2"),
    "three": Decimal("3"),
    "four": Decimal("4"),
    "five": Decimal("5"),
    "six": Decimal("6"),
    "seven": Decimal("7"),
    "eight": Decimal("8"),
    "nine": Decimal("9"),
    "ten": Decimal("10"),
    "eleven": Decimal("11"),
    "twelve": Decimal("12"),
    "thirteen": Decimal("13"),
    "fourteen": Decimal("14"),
    "fifteen": Decimal("15"),
    "sixteen": Decimal("16"),
    "seventeen": Decimal("17"),
    "eighteen": Decimal("18"),
    "nineteen": Decimal("19"),
    "twenty": Decimal("20"),
    "twenty one": Decimal("21"),
    "twenty-one": Decimal("21"),
    "twenty two": Decimal("22"),
    "twenty-two": Decimal("22"),
    "twenty three": Decimal("23"),
    "twenty-three": Decimal("23"),
    "twenty four": Decimal("24"),
    "twenty-four": Decimal("24"),
    "half": Decimal("0.5"),
}

DATE_FORMATS = (
    "%Y%m%d",
    "%d%m%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%Y.%m.%d",
)


def suggest_for_issue(issue: Mapping[str, Any]) -> IssueSuggestion | None:
    issue_type = str(issue.get("issue_type") or "")
    raw_value = _optional_string(issue.get("raw_value"))

    suggester = ISSUE_SUGGESTERS.get(issue_type)
    if suggester is None:
        return manual_check_suggestion(issue_type, raw_value)

    suggestion = suggester(raw_value)
    if suggestion is None:
        return manual_check_suggestion(issue_type, raw_value)

    return suggestion


def suggest_row_for_issue(issue: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    Return one dictionary shaped for insertion into validation_issue_suggestions.
    """
    suggestion = suggest_for_issue(issue)
    if suggestion is None:
        return None

    row = suggestion.to_dict()
    row["validation_issue_id"] = issue.get("id")
    row["import_run_id"] = issue.get("import_run_id")
    return row


def suggest_non_numeric_hours(raw_value: str | None) -> IssueSuggestion | None:
    value = _clean(raw_value)
    if not value:
        return None

    reasons: list[str] = []
    candidate = value

    hours_suffix = re.fullmatch(
        r"([+-]?\d+(?:[.,]\d+)?)\s*(?:h|hr|hrs|hour|hours)", candidate, re.IGNORECASE
    )
    if hours_suffix:
        candidate = hours_suffix.group(1)
        reasons.append("Removed an hours unit suffix.")

    if re.fullmatch(r"[+-]?\d+,\d+", candidate):
        candidate = candidate.replace(",", ".")
        reasons.append("Converted comma decimal separator to a dot decimal separator.")

    parsed = _parse_decimal(candidate)
    if parsed is not None and _valid_hours(parsed):
        confidence = "0.950" if len(reasons) == 1 else "0.900"
        return _suggestion(
            "non_numeric_hours",
            raw_value,
            parsed,
            _join_reasons(reasons),
            confidence,
        )

    word_value = NUMBER_WORDS.get(value.lower())
    if word_value is not None and _valid_hours(word_value):
        return _suggestion(
            "non_numeric_hours",
            raw_value,
            word_value,
            "Matched a written number to a numeric hours value.",
            "0.850",
        )

    half_match = re.fullmatch(r"(.+)\s+and\s+a\s+half", value.lower())
    if half_match:
        base = NUMBER_WORDS.get(half_match.group(1))
        if base is not None:
            parsed = base + Decimal("0.5")
            if _valid_hours(parsed):
                return _suggestion(
                    "non_numeric_hours",
                    raw_value,
                    parsed,
                    "Matched a written half-hour expression to a numeric value.",
                    "0.800",
                )

    return None


def suggest_hours_out_of_range(raw_value: str | None) -> IssueSuggestion | None:
    value = _clean(raw_value)
    parsed = _parse_decimal(value)
    if parsed is None:
        return None

    positive_value = abs(parsed)
    if parsed < 0 and _valid_hours(positive_value):
        return _suggestion(
            "hours_out_of_range",
            raw_value,
            positive_value,
            "Removed a negative sign because hours are expected to be positive.",
            "0.700",
        )

    divided_by_ten = parsed / Decimal("10")
    if parsed > 24 and _valid_hours(divided_by_ten):
        return _suggestion(
            "hours_out_of_range",
            raw_value,
            divided_by_ten,
            "Interpreted the value as missing a decimal separator.",
            "0.550",
        )

    return None


def suggest_invalid_date(raw_value: str | None) -> IssueSuggestion | None:
    value = _clean(raw_value)
    if not value:
        return None

    candidates: dict[str, tuple[str, str]] = {}

    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, date_format).date()
        except ValueError:
            continue

        confidence = "0.950"
        reason = "Parsed the date and converted it to YYYY-MM-DD format."
        if date_format in {"%m/%d/%Y", "%d/%m/%Y"}:
            confidence = "0.650"
            reason = "Parsed an ambiguous slash-formatted date; review before applying."

        candidates[parsed.isoformat()] = (reason, confidence)

    if len(candidates) != 1:
        return None

    suggested_value, (reason, confidence) = next(iter(candidates.items()))
    return IssueSuggestion(
        issue_type="invalid_date",
        raw_value=raw_value,
        suggested_value=suggested_value,
        suggestion_reason=reason,
        confidence=Decimal(confidence),
    )


def suggest_invalid_project_id_format(raw_value: str | None) -> IssueSuggestion | None:
    value = _clean(raw_value)
    if not value:
        return None

    normalized = value.upper().replace("_", "-").replace(" ", "-")

    exact_with_case_or_separator = re.fullmatch(r"PRJ-(\d{3})", normalized)
    if exact_with_case_or_separator:
        return IssueSuggestion(
            issue_type="invalid_project_id_format",
            raw_value=raw_value,
            suggested_value=f"PRJ-{exact_with_case_or_separator.group(1)}",
            suggestion_reason="Normalized project id casing or separators.",
            confidence=Decimal("0.950"),
        )

    prj_with_digits = re.fullmatch(r"PRJ-?(\d{1,3})", normalized)
    if prj_with_digits:
        return IssueSuggestion(
            issue_type="invalid_project_id_format",
            raw_value=raw_value,
            suggested_value=f"PRJ-{int(prj_with_digits.group(1)):03d}",
            suggestion_reason="Padded project id digits to the expected PRJ-XXX format.",
            confidence=Decimal("0.850"),
        )

    digits_only = re.fullmatch(r"(\d{1,3})", value)
    if digits_only:
        return IssueSuggestion(
            issue_type="invalid_project_id_format",
            raw_value=raw_value,
            suggested_value=f"PRJ-{int(digits_only.group(1)):03d}",
            suggestion_reason="Added the PRJ prefix and padded digits.",
            confidence=Decimal("0.800"),
        )

    return None


def suggest_duplicate_row(raw_value: str | None) -> IssueSuggestion:
    return IssueSuggestion(
        issue_type="duplicate_row",
        raw_value=raw_value,
        suggested_value=None,
        suggestion_reason="Potential duplicate row. Review manually before deleting or ignoring it.",
        confidence=Decimal("0.600"),
    )


def manual_check_suggestion(issue_type: str, raw_value: str | None) -> IssueSuggestion:
    return IssueSuggestion(
        issue_type=issue_type,
        raw_value=raw_value,
        suggested_value=None,
        suggestion_reason="Manual check needed. No automatic suggestion rule matched this value.",
        confidence=Decimal("0.000"),
    )


IssueSuggester = Callable[[str | None], IssueSuggestion | None]

ISSUE_SUGGESTERS: dict[str, IssueSuggester] = {
    "invalid_date": suggest_invalid_date,
    "non_numeric_hours": suggest_non_numeric_hours,
    "hours_out_of_range": suggest_hours_out_of_range,
    "invalid_project_id_format": suggest_invalid_project_id_format,
    "duplicate_row": suggest_duplicate_row,
}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def _parse_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _valid_hours(value: Decimal) -> bool:
    return Decimal("0") <= value <= Decimal("24")


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")


def _join_reasons(reasons: list[str]) -> str:
    return " ".join(reasons) if reasons else "Parsed as a numeric hours value."


def _suggestion(
    issue_type: str,
    raw_value: str | None,
    suggested_value: Decimal,
    suggestion_reason: str,
    confidence: str,
) -> IssueSuggestion:
    return IssueSuggestion(
        issue_type=issue_type,
        raw_value=raw_value,
        suggested_value=_format_decimal(suggested_value),
        suggestion_reason=suggestion_reason,
        confidence=Decimal(confidence),
    )
