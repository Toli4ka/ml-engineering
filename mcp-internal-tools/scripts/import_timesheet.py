from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_database_url
from timesheet_validation import validate_timesheet_csv


def file_sha256(path: Path) -> str:
    """Create SHA256 hash fingerprint"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        # read file by 1Mb to save memory
        for chunk in iter(lambda: fh.read(1024 * 1024), b""): 
            digest.update(chunk)
    return digest.hexdigest()


def import_timesheet(csv_path: Path, database_url: str) -> dict[str, Any]:
    validation = validate_timesheet_csv(csv_path)
    rows = validation.rows
    issues = validation.issues
    entry_ids_by_row_number: dict[int, int] = {}

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into import_runs (source_path, file_sha256)
                values (%s, %s)
                returning id
                """,
                (str(csv_path), file_sha256(csv_path)),
            )
            import_run_id = cur.fetchone()[0]

            for row in rows:
                cur.execute(
                    """
                    insert into timesheet_entries (
                        import_run_id,
                        source_row_number,
                        employee_id,
                        employee_name,
                        raw_date,
                        work_date,
                        project_id,
                        raw_hours_logged,
                        hours_logged
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    returning id
                    """,
                    (
                        import_run_id,
                        row.source_row_number,
                        row.employee_id,
                        row.employee_name,
                        row.raw_date,
                        row.work_date,
                        row.project_id,
                        row.raw_hours_logged,
                        row.hours_logged,
                    ),
                )
                entry_ids_by_row_number[row.source_row_number] = cur.fetchone()[0]

            for issue in issues:
                cur.execute(
                    """
                    insert into validation_issues (
                        import_run_id,
                        timesheet_entry_id,
                        source_row_number,
                        column_name,
                        issue_type,
                        severity,
                        raw_value,
                        message
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        import_run_id,
                        entry_ids_by_row_number.get(issue["source_row_number"]),
                        issue["source_row_number"],
                        issue["column_name"],
                        issue["issue_type"],
                        issue["severity"],
                        issue["raw_value"],
                        issue["message"],
                    ),
                )

    return {
        "import_run_id": str(import_run_id),
        "entry_count": len(rows),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a timesheet CSV into Postgres.")
    parser.add_argument("csv_path", type=Path, help="Path to the timesheet CSV file.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection URL. Defaults to DATABASE_URL from the environment or .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    result = import_timesheet(csv_path, args.database_url or get_database_url())
    print(f"Imported {result['entry_count']} rows into import_run {result['import_run_id']}")
    print(
        "Validation issues: "
        f"{result['issue_count']} total, "
        f"{result['error_count']} errors, "
        f"{result['warning_count']} warnings"
    )


if __name__ == "__main__":
    main()
