from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_database_url
from validation_issue_suggestions import suggest_row_for_issue


def fetch_validation_issues(
    import_run_id: str,
    database_url: str,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    filters = ["import_run_id = %s"]
    params: list[Any] = [import_run_id]

    if severity:
        filters.append("severity = %s")
        params.append(severity)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select
                    id,
                    import_run_id,
                    source_row_number,
                    column_name,
                    issue_type,
                    severity,
                    raw_value,
                    message
                from validation_issues
                where {' and '.join(filters)}
                order by source_row_number, id
                """,
                tuple(params),
            )
            return list(cur.fetchall())


def insert_suggestion(cur: Any, suggestion: dict[str, Any]) -> bool:
    cur.execute(
        """
        insert into validation_issue_suggestions (
            import_run_id,
            validation_issue_id,
            issue_type,
            raw_value,
            suggested_value,
            suggestion_reason,
            confidence
        )
        select %s, %s, %s, %s, %s, %s, %s
        where not exists (
            select 1
            from validation_issue_suggestions
            where validation_issue_id = %s
        )
        """,
        (
            suggestion["import_run_id"],
            suggestion["validation_issue_id"],
            suggestion["issue_type"],
            suggestion["raw_value"],
            suggestion["suggested_value"],
            suggestion["suggestion_reason"],
            suggestion["confidence"],
            suggestion["validation_issue_id"],
        ),
    )
    return cur.rowcount == 1


def generate_issue_suggestions(
    import_run_id: str,
    database_url: str,
    severity: str | None = None,
) -> dict[str, int]:
    issues = fetch_validation_issues(import_run_id, database_url, severity=severity)
    suggestions_generated = 0
    suggestions_inserted = 0

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for issue in issues:
                suggestion = suggest_row_for_issue(issue)
                if suggestion is None:
                    continue

                suggestions_generated += 1

                if insert_suggestion(cur, suggestion):
                    suggestions_inserted += 1

    return {
        "issues_checked": len(issues),
        "suggestions_generated": suggestions_generated,
        "suggestions_inserted": suggestions_inserted,
        "suggestions_skipped_as_duplicates": suggestions_generated - suggestions_inserted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fix suggestions for stored validation issues."
    )
    parser.add_argument("import_run_id", help="Import run id to generate suggestions for.")
    parser.add_argument(
        "--severity",
        choices=["error", "warning"],
        help="Only generate suggestions for issues with this severity.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection URL. Defaults to DATABASE_URL from the environment or .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_issue_suggestions(
        import_run_id=args.import_run_id,
        database_url=args.database_url or get_database_url(),
        severity=args.severity,
    )

    print(f"Checked {result['issues_checked']} validation issues.")
    print(f"Generated suggestions: {result['suggestions_generated']}")
    print(f"Inserted new suggestions: {result['suggestions_inserted']}")
    print(f"Skipped duplicates: {result['suggestions_skipped_as_duplicates']}")


if __name__ == "__main__":
    main()
