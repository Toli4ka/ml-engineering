from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) #TODO: why this line? 

from timesheet_validation import validate_timesheet_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a timesheet CSV without touching Postgres.")
    parser.add_argument("csv_path", type=Path, help="Path to the timesheet CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    result = validate_timesheet_csv(csv_path)
    print(f"Validated {len(result.rows)} rows from {csv_path}")
    print(
        "Validation issues: "
        f"{len(result.issues)} total, "
        f"{result.error_count} errors, "
        f"{result.warning_count} warnings"
    )
    for issue in result.issues:
        print(
            f"row {issue['source_row_number']}: "
            f"{issue['severity']} {issue['issue_type']} - {issue['message']}"
        )


if __name__ == "__main__":
    main()
