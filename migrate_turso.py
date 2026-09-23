import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine

from turso_config import (
    turso_connect_args,
    turso_is_configured,
    turso_sqlalchemy_uri,
)


ROOT = Path(__file__).resolve().parent
SOURCE_DB = ROOT / "instance" / "coins.db"
MIGRATION_NAME = "initial_sqlite_import_v1"
MIGRATION_TABLE = "__coin_collection_migration"


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def create_if_not_exists(ddl):
    return re.sub(
        r"^\s*CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS)",
        "CREATE TABLE IF NOT EXISTS ",
        ddl,
        count=1,
        flags=re.IGNORECASE,
    )


def index_if_not_exists(ddl):
    return re.sub(
        r"^\s*CREATE\s+(UNIQUE\s+)?INDEX\s+(?!IF\s+NOT\s+EXISTS)",
        lambda match: "CREATE " + (match.group(1) or "") + "INDEX IF NOT EXISTS ",
        ddl,
        count=1,
        flags=re.IGNORECASE,
    )


def source_tables(source):
    rows = source.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY rowid
        """
    ).fetchall()

    return [
        (name, ddl)
        for name, ddl in rows
        if name != MIGRATION_TABLE and ddl
    ]


def table_exists(conn, name):
    row = conn.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).first()
    return row is not None


def target_count(conn, table_name):
    return conn.exec_driver_sql(
        f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
    ).scalar_one()


def source_count(source, table_name):
    return source.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
    ).fetchone()[0]


def ensure_target_schema(engine, source, tables):
    with engine.begin() as conn:
        for _, ddl in tables:
            conn.exec_driver_sql(create_if_not_exists(ddl))

        index_rows = source.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND sql IS NOT NULL
            ORDER BY rowid
            """
        ).fetchall()

        for (ddl,) in index_rows:
            conn.exec_driver_sql(index_if_not_exists(ddl))

        conn.exec_driver_sql(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                name TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def get_migration_state(engine):
    with engine.connect() as conn:
        row = conn.exec_driver_sql(
            f"SELECT state FROM {MIGRATION_TABLE} WHERE name=?",
            (MIGRATION_NAME,),
        ).first()
        return row[0] if row else None


def set_migration_state(engine, state):
    now = datetime.now(timezone.utc).isoformat()

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            INSERT INTO {MIGRATION_TABLE} (name, state, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                state=excluded.state,
                updated_at=excluded.updated_at
            """,
            (MIGRATION_NAME, state, now),
        )


def verify_counts(engine, source, tables):
    mismatches = []

    with engine.connect() as conn:
        for table_name, _ in tables:
            expected = source_count(source, table_name)
            actual = target_count(conn, table_name)

            print(f"Turso verify {table_name}: {actual}/{expected}", flush=True)

            if actual != expected:
                mismatches.append((table_name, expected, actual))

    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected}, got {actual}"
            for name, expected, actual in mismatches
        )
        raise RuntimeError(f"Turso row-count verification failed: {details}")


def clear_target(engine, tables):
    with engine.begin() as conn:
        try:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        except Exception:
            pass

        for table_name, _ in reversed(tables):
            if table_exists(conn, table_name):
                conn.exec_driver_sql(
                    f"DELETE FROM {quote_identifier(table_name)}"
                )


def copy_table(engine, source, table_name):
    info = source.execute(
        f"PRAGMA table_info({quote_identifier(table_name)})"
    ).fetchall()

    columns = [row[1] for row in info]

    if not columns:
        return 0

    column_sql = ", ".join(quote_identifier(name) for name in columns)
    placeholders = ", ".join("?" for _ in columns)

    rows = source.execute(
        f"SELECT {column_sql} FROM {quote_identifier(table_name)}"
    ).fetchall()

    if not rows:
        return 0

    insert_sql = (
        f"INSERT INTO {quote_identifier(table_name)} "
        f"({column_sql}) VALUES ({placeholders})"
    )

    chunk_size = 200

    with engine.begin() as conn:
        for start in range(0, len(rows), chunk_size):
            chunk = [
                tuple(row)
                for row in rows[start:start + chunk_size]
            ]
            conn.exec_driver_sql(insert_sql, chunk)

    return len(rows)


def main():
    if not turso_is_configured():
        raise SystemExit("Turso variables are not configured.")

    if not SOURCE_DB.is_file():
        raise SystemExit(f"Source SQLite database was not found: {SOURCE_DB}")

    source = sqlite3.connect(str(SOURCE_DB))

    try:
        tables = source_tables(source)

        if not tables:
            raise RuntimeError("No application tables were found in the source SQLite database.")

        engine = create_engine(
            turso_sqlalchemy_uri(),
            connect_args=turso_connect_args(),
            pool_pre_ping=True,
        )

        ensure_target_schema(engine, source, tables)

        state = get_migration_state(engine)

        if state == "complete":
            print("Turso migration marker is already complete; verifying counts.", flush=True)
            verify_counts(engine, source, tables)
            print("Turso migration already complete and verified.", flush=True)
            return

        if state is None:
            with engine.connect() as conn:
                nonempty = []

                for table_name, _ in tables:
                    if table_exists(conn, table_name):
                        count = target_count(conn, table_name)
                        if count:
                            nonempty.append((table_name, count))

            if nonempty:
                details = ", ".join(
                    f"{name}={count}"
                    for name, count in nonempty
                )
                raise RuntimeError(
                    "Turso already contains application data without our migration marker; "
                    f"refusing to overwrite it ({details})."
                )

            set_migration_state(engine, "in_progress")

        print("Starting SQLite -> Turso migration.", flush=True)
        clear_target(engine, tables)

        total_rows = 0

        for table_name, _ in tables:
            copied = copy_table(engine, source, table_name)
            total_rows += copied
            print(f"Turso copied {table_name}: {copied} rows", flush=True)

        verify_counts(engine, source, tables)
        set_migration_state(engine, "complete")

        print(
            f"Turso migration complete: {len(tables)} tables, {total_rows} rows verified.",
            flush=True,
        )

    finally:
        source.close()


if __name__ == "__main__":
    main()
