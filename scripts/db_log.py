#!/usr/bin/env python3
"""db_log.py — write rows into the provenance/reproducibility registry.

Companion to db/schema.sql. Stdlib sqlite3 only; no third-party deps.

Subcommands:
  init      create/upgrade the DB by executing db/schema.sql
  dataset   insert (or upsert) one datasets row from --json and/or flags
  run       insert (or upsert) one runs row from --json and/or flags

Row data may come from a JSON object (--json FILE, or --json - for stdin) and/or
individual --field flags; flags override JSON keys. Unknown keys are rejected
fail-closed so a typo never silently vanishes. Foreign keys are enforced on
every connection (SQLite defaults them OFF).

Usage:
  python scripts/db_log.py init --db db/registry.sqlite
  python scripts/db_log.py dataset --db db/registry.sqlite \
      --dataset-id corpus-v3 --name "neoperi passages" \
      --license CC-BY --commercial-clean 1 --role corpus
  python scripts/db_log.py run --db db/registry.sqlite --json run.json
  echo '{"run_id":"r42","run_type":"train"}' | \
      python scripts/db_log.py run --db db/registry.sqlite --json -
"""
import argparse
import json
import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SCHEMA = os.path.join(_HERE, "..", "db", "schema.sql")

# Columns we accept per table. Order is not significant; the PK is required.
DATASET_COLS = (
    "dataset_id", "name", "source_url", "license", "commercial_clean",
    "gated", "role", "est_rows", "content_hash", "retrieved_at",
)
RUN_COLS = (
    "run_id", "run_type", "student_model_id", "student_version_date",
    "teacher_model_id", "teacher_version_date", "phase", "peft", "lora_r",
    "lora_alpha", "seq_len", "hyperparams_json", "dataset_hash",
    "heldout_set_hash", "git_commit", "seed", "started_at", "finished_at",
    "status",
)

# Flag name (dashes) -> column name (underscores) is a pure translation, so we
# derive flags from the column tuples and only special-case the JSON payload.


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _load_json(path):
    if not path:
        return {}
    raw = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"ABORT: --json is not valid JSON: {e}")
    if not isinstance(obj, dict):
        sys.exit("ABORT: --json must be a single JSON object.")
    return obj


def _collect(args, cols, pk):
    """Merge --json payload with per-column flags; flags win. Fail closed."""
    row = _load_json(getattr(args, "json", None))
    bad = [k for k in row if k not in cols]
    if bad:
        sys.exit(f"ABORT: --json has unknown keys for this table: {bad}")
    for col in cols:
        val = getattr(args, col, None)   # argparse stored flags under dest=col
        if val is not None:
            row[col] = val
    # hyperparams_json may be handed in as a dict/list — serialize it.
    if "hyperparams_json" in row and not isinstance(row["hyperparams_json"], str):
        row["hyperparams_json"] = json.dumps(row["hyperparams_json"],
                                             ensure_ascii=False)
    if not row.get(pk):
        sys.exit(f"ABORT: '{pk}' is required (via --json or --{pk.replace('_', '-')}).")
    return row


def _upsert(conn, table, row, pk):
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != pk)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT({pk}) DO UPDATE SET {updates}")
    try:
        conn.execute(sql, [row[c] for c in cols])
        conn.commit()
    except sqlite3.Error as e:
        sys.exit(f"ABORT: insert into {table} failed: {e}")


def cmd_init(args):
    schema = args.schema or _DEFAULT_SCHEMA
    if not os.path.exists(schema):
        sys.exit(f"ABORT: schema file not found: {schema}")
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    with open(schema, encoding="utf-8") as fh:
        ddl = fh.read()
    conn = _connect(args.db)
    try:
        conn.executescript(ddl)
        conn.commit()
    except sqlite3.Error as e:
        sys.exit(f"ABORT: applying schema failed: {e}")
    finally:
        conn.close()
    print(f"OK: initialized {args.db} from {schema}")


def cmd_dataset(args):
    row = _collect(args, DATASET_COLS, "dataset_id")
    conn = _connect(args.db)
    try:
        _upsert(conn, "datasets", row, "dataset_id")
    finally:
        conn.close()
    print(f"OK: datasets <- {row['dataset_id']}")


def cmd_run(args):
    row = _collect(args, RUN_COLS, "run_id")
    conn = _connect(args.db)
    try:
        _upsert(conn, "runs", row, "run_id")
    finally:
        conn.close()
    print(f"OK: runs <- {row['run_id']}")


def _add_col_flags(sub, cols):
    """One --column-name flag per table column (dest keeps the underscore name)."""
    for col in cols:
        sub.add_argument(f"--{col.replace('_', '-')}", dest=col, default=None)


def main():
    # --db is shared by every subcommand and accepted after the verb (as the
    # docstring shows) via a parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=os.path.join(_HERE, "..", "db", "registry.sqlite"),
                        help="path to the SQLite registry (default db/registry.sqlite)")

    ap = argparse.ArgumentParser(description="Log rows into the neoperi provenance registry.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", parents=[common],
                            help="create/upgrade the DB from schema.sql")
    p_init.add_argument("--schema", default=None, help="path to schema.sql")
    p_init.set_defaults(func=cmd_init)

    p_ds = sub.add_parser("dataset", parents=[common], help="insert/upsert a datasets row")
    p_ds.add_argument("--json", default=None, help="JSON object file, or - for stdin")
    _add_col_flags(p_ds, DATASET_COLS)
    p_ds.set_defaults(func=cmd_dataset)

    p_run = sub.add_parser("run", parents=[common], help="insert/upsert a runs row")
    p_run.add_argument("--json", default=None, help="JSON object file, or - for stdin")
    _add_col_flags(p_run, RUN_COLS)
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
