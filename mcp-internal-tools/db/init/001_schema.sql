create extension if not exists pgcrypto;

create table if not exists import_runs (
    id uuid primary key default gen_random_uuid(),
    source_path text not null,
    file_sha256 text,
    imported_at timestamptz not null default now()
);

create table if not exists timesheet_entries (
    id bigserial primary key,
    import_run_id uuid not null references import_runs(id) on delete cascade,
    source_row_number integer not null,
    employee_id text not null,
    employee_name text not null,
    raw_date text not null,
    work_date date,
    project_id text not null,
    raw_hours_logged text not null,
    hours_logged numeric,
    created_at timestamptz not null default now()
);

create table if not exists validation_issues (
    id bigserial primary key,
    import_run_id uuid not null references import_runs(id) on delete cascade,
    timesheet_entry_id bigint references timesheet_entries(id) on delete cascade,
    source_row_number integer,
    column_name text,
    issue_type text not null,
    severity text not null,
    raw_value text,
    message text not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_timesheet_entries_import_run_id
    on timesheet_entries(import_run_id);

create index if not exists idx_timesheet_entries_employee_id
    on timesheet_entries(employee_id);

create index if not exists idx_timesheet_entries_project_id
    on timesheet_entries(project_id);

create index if not exists idx_validation_issues_import_run_id
    on validation_issues(import_run_id);

create index if not exists idx_validation_issues_severity
    on validation_issues(severity);
