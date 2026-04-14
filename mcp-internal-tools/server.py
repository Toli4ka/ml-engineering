from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP
import psycopg
from psycopg.rows import dict_row

from config import get_database_url

# FastMCP auto-generates tool schemas from type hints/docstrings.
mcp = FastMCP("Timesheet DB Reader", json_response=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, UUID):
        return str(value)
    return value


def _fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with psycopg.connect(get_database_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [
                {key: _json_value(value) for key, value in row.items()}
                for row in cur.fetchall()
            ]


def _fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = _fetch_all(query, params)
    return rows[0] if rows else None


@mcp.tool()
def list_import_runs(limit: int = 20) -> dict[str, Any]:
    """
    List recent CSV import runs from Postgres.

    This tool is read-only and does not validate CSV files.
    """
    safe_limit = max(1, min(limit, 100))
    rows = _fetch_all(
        """
        select
            id,
            source_path,
            file_sha256,
            imported_at
        from import_runs
        order by imported_at desc
        limit %s
        """,
        (safe_limit,),
    )
    return {"status": "ok", "import_runs": rows}


@mcp.tool()
def get_import_summary(import_run_id: str) -> dict[str, Any]:
    """
    Return row and issue counts for one import run.

    This tool is read-only and expects an existing import_run_id.
    """
    summary = _fetch_one(
        """
        select
            r.id as import_run_id,
            r.source_path,
            r.imported_at,
            count(distinct e.id) as entry_count,
            count(distinct i.id) as issue_count,
            count(distinct i.id) filter (where i.severity = 'error') as error_count,
            count(distinct i.id) filter (where i.severity = 'warning') as warning_count
        from import_runs r
        left join timesheet_entries e
            on e.import_run_id = r.id
        left join validation_issues i
            on i.import_run_id = r.id
        where r.id = %s
        group by r.id, r.source_path, r.imported_at
        """,
        (import_run_id,),
    )
    if summary is None:
        return {"status": "error", "message": f"Import run not found: {import_run_id}"}
    return {"status": "ok", "summary": summary}


@mcp.tool()
def list_timesheet_entries(
    import_run_id: str,
    employee_id: str | None = None,
    project_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    List imported timesheet rows from Postgres.

    Optional filters can narrow by employee_id or project_id.
    """
    safe_limit = max(1, min(limit, 200))
    filters = ["import_run_id = %s"]
    params: list[Any] = [import_run_id]

    if employee_id:
        filters.append("employee_id = %s")
        params.append(employee_id)
    if project_id:
        filters.append("project_id = %s")
        params.append(project_id)

    params.append(safe_limit)
    rows = _fetch_all(
        f"""
        select
            id,
            source_row_number,
            employee_id,
            employee_name,
            raw_date,
            work_date,
            project_id,
            raw_hours_logged,
            hours_logged,
            created_at
        from timesheet_entries
        where {' and '.join(filters)}
        order by source_row_number
        limit %s
        """,
        tuple(params),
    )
    return {"status": "ok", "entries": rows, "returned_count": len(rows)}


@mcp.tool()
def list_validation_issues(
    import_run_id: str,
    severity: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    List validation issues already stored in Postgres.

    This tool does not run validation; it only reads previously imported issues.
    """
    safe_limit = max(1, min(limit, 200))
    filters = ["i.import_run_id = %s"]
    params: list[Any] = [import_run_id]

    if severity:
        filters.append("i.severity = %s")
        params.append(severity)

    params.append(safe_limit)
    rows = _fetch_all(
        f"""
        select
            i.id,
            i.source_row_number,
            i.column_name,
            i.issue_type,
            i.severity,
            i.raw_value,
            i.message,
            e.employee_id,
            e.employee_name,
            e.project_id
        from validation_issues i
        left join timesheet_entries e
            on e.id = i.timesheet_entry_id
        where {' and '.join(filters)}
        order by i.source_row_number, i.id
        limit %s
        """,
        tuple(params),
    )
    return {"status": "ok", "issues": rows, "returned_count": len(rows)}


@mcp.tool()
def get_hours_by_employee(import_run_id: str) -> dict[str, Any]:
    """
    Summarize parsed hours by employee for one import run.

    Rows with non-numeric hours are ignored by the sum because hours_logged is null.
    """
    rows = _fetch_all(
        """
        select
            employee_id,
            employee_name,
            count(*) as row_count,
            sum(hours_logged) as total_hours
        from timesheet_entries
        where import_run_id = %s
        group by employee_id, employee_name
        order by employee_name, employee_id
        """,
        (import_run_id,),
    )
    return {"status": "ok", "hours_by_employee": rows}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
