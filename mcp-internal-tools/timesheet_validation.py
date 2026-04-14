from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any


REQUIRED_COLUMNS = [
    "employee_id",
    "employee_name",
    "date",
    "project_id",
    "hours_logged",
]
PROJECT_ID_PATTERN = re.compile(r"PRJ-\d{3}")
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d")


@dataclass(frozen=True)
class PreparedRow:
    source_row_number: int
    employee_id: str
    employee_name: str
    raw_date: str
    work_date: str | None
    project_id: str
    raw_hours_logged: str
    hours_logged: Decimal | None


@dataclass(frozen=True)
class ValidationResult:
    rows: list[PreparedRow]
    issues: list[dict[str, Any]]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue["severity"] == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue["severity"] == "warning")


def parse_date(raw_value: str) -> str | None:
    value = raw_value.strip()
    if not value:
        return None

    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            pass

    return None


def parse_hours(raw_value: str) -> Decimal | None:
    value = raw_value.strip()
    if not value:
        return None

    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def load_csv_rows(csv_path: Path) -> list[PreparedRow]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        rows: list[PreparedRow] = []
        for index, row in enumerate(reader, start=2):
            raw_date = (row["date"] or "").strip()
            raw_hours = (row["hours_logged"] or "").strip()
            rows.append(
                PreparedRow(
                    source_row_number=index,
                    employee_id=(row["employee_id"] or "").strip(),
                    employee_name=(row["employee_name"] or "").strip(),
                    raw_date=raw_date,
                    work_date=parse_date(raw_date),
                    project_id=(row["project_id"] or "").strip(),
                    raw_hours_logged=raw_hours,
                    hours_logged=parse_hours(raw_hours),
                )
            )

    return rows


def build_issues(rows: list[PreparedRow]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    duplicate_keys = Counter(
        (row.employee_id, row.raw_date, row.project_id, row.raw_hours_logged)
        for row in rows
    )

    for row in rows:
        if row.work_date is None:
            issues.append(
                {
                    "source_row_number": row.source_row_number,
                    "column_name": "date",
                    "issue_type": "invalid_date",
                    "severity": "error",
                    "raw_value": row.raw_date,
                    "message": f"Invalid date: {row.raw_date!r}",
                }
            )

        if row.hours_logged is None:
            issues.append(
                {
                    "source_row_number": row.source_row_number,
                    "column_name": "hours_logged",
                    "issue_type": "non_numeric_hours",
                    "severity": "error",
                    "raw_value": row.raw_hours_logged,
                    "message": f"Non-numeric hours_logged: {row.raw_hours_logged!r}",
                }
            )
        elif row.hours_logged < 0 or row.hours_logged > 24:
            issues.append(
                {
                    "source_row_number": row.source_row_number,
                    "column_name": "hours_logged",
                    "issue_type": "hours_out_of_range",
                    "severity": "error",
                    "raw_value": row.raw_hours_logged,
                    "message": f"hours_logged must be between 0 and 24, got {row.hours_logged}",
                }
            )

        if not PROJECT_ID_PATTERN.fullmatch(row.project_id):
            issues.append(
                {
                    "source_row_number": row.source_row_number,
                    "column_name": "project_id",
                    "issue_type": "invalid_project_id_format",
                    "severity": "error",
                    "raw_value": row.project_id,
                    "message": f"Invalid project_id format: {row.project_id!r}. Expected PRJ-XXX.",
                }
            )

        duplicate_key = (row.employee_id, row.raw_date, row.project_id, row.raw_hours_logged)
        if duplicate_keys[duplicate_key] > 1:
            issues.append(
                {
                    "source_row_number": row.source_row_number,
                    "column_name": "employee_id,date,project_id,hours_logged",
                    "issue_type": "duplicate_row",
                    "severity": "warning",
                    "raw_value": str(row.source_row_number),
                    "message": "Potential duplicate timesheet row.",
                }
            )

    return issues


def validate_timesheet_csv(csv_path: Path) -> ValidationResult:
    rows = load_csv_rows(csv_path)
    return ValidationResult(rows=rows, issues=build_issues(rows))
