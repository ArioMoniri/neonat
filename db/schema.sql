-- schema.sql — provenance + reproducibility registry for the neoperi CDSS.
-- ============================================================================
-- SQLite-compatible DDL. One place to answer, for any generated card or metric:
--   which dataset / passage did it come from, under what license, from which
--   run (student+teacher versions, PEFT config, git commit, seed), and how did
--   it score. RESEARCH prototype over SYNTHETIC data — never clinically released.
--
-- Apply with:  sqlite3 db/registry.sqlite < db/schema.sql
--         or:  python scripts/db_log.py init --db db/registry.sqlite
--
-- Design notes:
--   * fail-closed CHECK constraints mirror the card discriminated union and the
--     read-only action-verb whitelist enforced in train_lora.py, so a bad row
--     is rejected at the DB boundary the same way validate_card rejects a card.
--   * foreign keys are declared; callers must `PRAGMA foreign_keys = ON;`
--     (db_log.py does this on every connection — SQLite defaults it OFF).
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- datasets — one row per corpus/source pulled into the pipeline.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    source_url       TEXT,
    license          TEXT,
    -- 1 => cleared for commercial use (permissive/own); 0 => NC/unknown/gated.
    commercial_clean INTEGER NOT NULL DEFAULT 0 CHECK (commercial_clean IN (0, 1)),
    gated            INTEGER NOT NULL DEFAULT 0 CHECK (gated IN (0, 1)),
    -- how this dataset is used in the pipeline.
    role             TEXT CHECK (role IN (
                         'train', 'eval', 'heldout', 'corpus', 'benchmark', 'mcq'
                     )),
    est_rows         INTEGER,
    content_hash     TEXT,               -- hash of the materialized artifact
    retrieved_at     TEXT                -- ISO-8601 UTC
);

-- ---------------------------------------------------------------------------
-- passages — grounding text units; the `kaynak` a grounded card cites.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS passages (
    passage_id      TEXT PRIMARY KEY,
    dataset_id      TEXT REFERENCES datasets(dataset_id),
    source          TEXT,
    text            TEXT,
    specialty       TEXT,               -- e.g. neonatology / perinatology
    language        TEXT DEFAULT 'tr',
    license         TEXT,
    -- 1 => No-Derivatives restricted (ND); blocks derivative training use.
    is_nd_restricted INTEGER NOT NULL DEFAULT 0 CHECK (is_nd_restricted IN (0, 1)),
    oldest_date     TEXT,               -- provenance date span of the content
    newest_date     TEXT,
    embedding_ref   TEXT                -- opaque ref into the vector store (HNSW)
);
CREATE INDEX IF NOT EXISTS idx_passages_dataset ON passages(dataset_id);

-- ---------------------------------------------------------------------------
-- runs — one row per training/eval invocation. The reproducibility anchor.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    run_type             TEXT CHECK (run_type IN (
                             'train', 'eval', 'benchmark', 'distill', 'synth'
                         )),
    student_model_id     TEXT,
    student_version_date TEXT,
    teacher_model_id     TEXT,
    teacher_version_date TEXT,
    phase                TEXT,           -- e.g. sft / dpo / cpt
    peft                 TEXT,           -- e.g. qlora-nf4+rslora+dora
    lora_r               INTEGER,
    lora_alpha           INTEGER,
    seq_len              INTEGER,
    hyperparams_json     TEXT,           -- full arg/hparam snapshot as JSON
    dataset_hash         TEXT,           -- ties back to datasets.content_hash
    heldout_set_hash     TEXT,
    git_commit           TEXT,
    seed                 INTEGER,
    started_at           TEXT,
    finished_at          TEXT,
    status               TEXT NOT NULL DEFAULT 'started' CHECK (status IN (
                             'started', 'running', 'finished', 'failed', 'aborted'
                         ))
);

-- ---------------------------------------------------------------------------
-- cases — vignettes / eval items. category + karar mirror the card union.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cases (
    case_id         TEXT PRIMARY KEY,
    dataset_id      TEXT REFERENCES datasets(dataset_id),
    -- provenance.category from the training/eval rows.
    category        TEXT CHECK (category IN (
                        'grounded', 'refusal', 'empty_passage',
                        'boundary_pressure', 'acuity'
                    )),
    vignette_tr     TEXT,
    passage_id      TEXT REFERENCES passages(passage_id),   -- nullable (refusal/empty)
    is_groundable   INTEGER CHECK (is_groundable IN (0, 1)),
    held_out        INTEGER NOT NULL DEFAULT 0 CHECK (held_out IN (0, 1)),
    split           TEXT CHECK (split IN ('train', 'val', 'test', 'heldout')),
    eval_set_hash   TEXT,
    reviewed        INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
    created_by_run_id TEXT REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_cases_dataset  ON cases(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cases_passage  ON cases(passage_id);
CREATE INDEX IF NOT EXISTS idx_cases_category ON cases(category);

-- ---------------------------------------------------------------------------
-- patient_states — agentic hasta_durumu snapshots, per case turn.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patient_states (
    state_id        INTEGER PRIMARY KEY,
    case_id         TEXT NOT NULL REFERENCES cases(case_id),
    turn_index      INTEGER NOT NULL DEFAULT 0,
    gebelik_haftasi REAL,               -- gestational age (weeks)
    postnatal_gun   INTEGER,            -- postnatal day
    aktif_problem   TEXT
);
CREATE INDEX IF NOT EXISTS idx_patient_states_case ON patient_states(case_id);

-- ---------------------------------------------------------------------------
-- actions — onerilen_eylem; verb is the read-only whitelist (train_lora.py).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actions (
    action_id       INTEGER PRIMARY KEY,
    case_id         TEXT NOT NULL REFERENCES cases(case_id),
    turn_index      INTEGER NOT NULL DEFAULT 0,
    verb            TEXT CHECK (verb IN (
                        'sor', 'gozlemle', 'gözlemle',
                        'tetkik_iste_degerlendirme_icin',
                        'hekime_danis', 'hekime_danış'
                    )),
    aciklama        TEXT,
    is_order        INTEGER NOT NULL DEFAULT 0 CHECK (is_order IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_actions_case ON actions(case_id);

-- ---------------------------------------------------------------------------
-- action_results — eylem_sonucu observed after an action.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS action_results (
    result_id           INTEGER PRIMARY KEY,
    action_id           INTEGER NOT NULL REFERENCES actions(action_id),
    gozlenen_sonuc      TEXT,
    durum_guncellemesi  TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_results_action ON action_results(action_id);

-- ---------------------------------------------------------------------------
-- eval_scores — one row per (run, case, metric) with optional CI.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_scores (
    score_id    INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    case_id     TEXT REFERENCES cases(case_id),
    metric      TEXT NOT NULL,
    value       REAL,
    ci_low      REAL,
    ci_high     REAL,
    grader      TEXT                    -- e.g. rule / llm-judge / human
);
CREATE INDEX IF NOT EXISTS idx_eval_scores_run    ON eval_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_metric ON eval_scores(metric);

-- ---------------------------------------------------------------------------
-- commercial_clean VIEW — datasets cleared for commercial/derivative use.
-- Fail-closed: only rows explicitly flagged clean AND not gated surface here.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS commercial_clean_datasets AS
    SELECT dataset_id, name, source_url, license, role, est_rows,
           content_hash, retrieved_at
    FROM datasets
    WHERE commercial_clean = 1 AND gated = 0;
