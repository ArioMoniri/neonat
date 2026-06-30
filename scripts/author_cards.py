#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author_cards.py — Human-in-the-loop dataset authoring for the TR neoperi CDSS.
================================================================================
This helper NEVER fabricates clinical content. It only:
  1. `template`  — write an example passages file so you know the input shape.
  2. `build`     — turn guideline passages into *draft* training rows
                   (reviewed:false, EMPTY suggestions) for a clinician to fill in.
  3. `promote`   — validate clinician-edited rows with the SAME fail-closed gate
                   as training, and append ONLY reviewed:true + grounded rows to
                   the training file, recording provenance.

Workflow:
    python author_cards.py template --out data/staging/passages.jsonl
    # (curate real guideline passages into that file)
    python author_cards.py build --passages data/staging/passages.jsonl \
                                 --out data/staging/drafts.jsonl
    # clinicians edit drafts.jsonl: fill onerilen_sorular/tetkikler from the
    # passage, set eksik_veriler, then flip "reviewed": true on approved rows
    python author_cards.py promote --in data/staging/drafts.jsonl \
                                   --out data/processed/task_sft.jsonl

The clinical content is authored and approved by humans; this tool only scaffolds
the format and enforces the gate. A row a clinician did not mark reviewed:true is
never promoted.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_train_module():
    """Import train_lora.py by path to reuse its canonical guardrail + validator."""
    path = os.path.join(_HERE, "train_lora.py")
    spec = importlib.util.spec_from_file_location("train_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TL = _load_train_module()
GUARDRAIL_SYSTEM = TL.GUARDRAIL_SYSTEM
DEFAULT_UYARI = ("Bu öneriler yalnızca verilen kılavuza dayanır ve klinik karar "
                 "yerine geçmez; nihai değerlendirme hekime aittir.")


# ----------------------------------------------------------------------------
def cmd_template(args):
    rows = [
        {"passage_id": "TND-Hiperbilirubinemi-2021-s4",
         "passage": "[Buraya ilgili kılavuz pasajının TAM metnini yapıştırın.]",
         "context": "[Hasta bağlamı şablonu: gebelik haftası, doğum ağırlığı, "
                    "postnatal gün, ilgili öykü vb.]"},
        {"passage_id": "[kaynak-kimligi-2]",
         "passage": "[ikinci kılavuz pasajı]",
         "context": "[ikinci hasta bağlamı]"},
    ]
    _write_jsonl(args.out, rows)
    print(f"==> Wrote {len(rows)} example passage rows to {args.out}")
    print("    Replace the bracketed placeholders with real, curated guideline text.")


def cmd_build(args):
    passages = _read_jsonl(args.passages)
    if not passages:
        sys.exit(f"ABORT: no passages found in {args.passages}")
    drafts = []
    for i, p in enumerate(passages, 1):
        pid = str(p.get("passage_id", "")).strip()
        passage = str(p.get("passage", "")).strip()
        context = str(p.get("context", "")).strip()
        if not pid or not passage:
            sys.exit(f"ABORT: passage row {i} needs non-empty 'passage_id' and 'passage'.")
        user = (f"Hasta bağlamı: {context or '[bağlam sağlanmadı]'}. "
                f"Kılavuz pasajı: {passage}")
        # DRAFT card: empty suggestions + EMPTY kaynak. The clinician fills the
        # suggestions from the passage AND sets kaynak themselves, so grounding is
        # an explicit human act (the passage id below is provided only as a hint).
        card = {
            "onerilen_sorular": [],     # <- clinician fills from the passage
            "onerilen_tetkikler": [],   # <- clinician fills from the passage
            "eksik_veriler": [],        # <- clinician lists pivotal absent data
            "kaynak": "",               # <- clinician sets when adding suggestions
            "uyari": DEFAULT_UYARI,     # hint: passage_id for this draft = {pid}
        }
        _ = pid  # passage_id is shown in the user turn; left out of the card on purpose
        drafts.append({
            "messages": [
                {"role": "system", "content": GUARDRAIL_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(card, ensure_ascii=False)},
            ],
            "reviewed": False,          # <- clinician flips to true once approved
        })
    _write_jsonl(args.out, drafts)
    print(f"==> Wrote {len(drafts)} DRAFT rows (reviewed:false) to {args.out}")
    print("    Clinicians: fill onerilen_sorular/onerilen_tetkikler from the passage,")
    print("    list pivotal missing data in eksik_veriler, then set reviewed:true.")


def cmd_promote(args):
    drafts = _read_jsonl(args.infile)
    if not drafts:
        sys.exit(f"ABORT: no rows in {args.infile}")
    approved, skipped, rejected = [], 0, []
    for i, obj in enumerate(drafts, 1):
        if obj.get("reviewed", False) is not True:
            skipped += 1            # not yet clinician-approved — leave in staging
            continue
        ok, reason, clean = TL._validate_row(obj)
        if not ok:
            rejected.append((i, reason))
            continue
        approved.append(clean)

    if rejected:
        preview = "\n".join(f"       row {ln}: {why}" for ln, why in rejected[:15])
        sys.exit(f"ABORT: {len(rejected)} reviewed row(s) FAILED the fail-closed "
                 f"gate. Fix them in staging; nothing was promoted.\n{preview}")
    if not approved:
        print(f"==> Nothing to promote ({skipped} row(s) still reviewed:false).")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Idempotency: dedup against rows already in the training file by content hash,
    # so re-running promote on the same staging file does not duplicate rows.
    existing = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    existing.add(_row_hash(json.loads(line)))

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = os.path.join(os.path.dirname(args.out) or ".", "manifest.csv")
    new_manifest = not os.path.exists(manifest)
    written, dup = 0, 0
    with open(args.out, "a", encoding="utf-8") as out_fh, \
            open(manifest, "a", newline="", encoding="utf-8") as man_fh:
        w = csv.writer(man_fh)
        if new_manifest:
            w.writerow(["timestamp_utc", "source_file", "row_sha256", "kaynak", "reviewer"])
        for r in approved:
            h = _row_hash(r)
            if h in existing:
                dup += 1
                continue
            existing.add(h)
            out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            card = json.loads(r["messages"][-1]["content"])
            w.writerow([ts, args.infile, h, card.get("kaynak", ""), args.reviewer or ""])
            written += 1

    print(f"==> Promoted {written} new reviewed+grounded row(s) -> {args.out}")
    if dup:
        print(f"    (skipped {dup} already-present duplicate row(s).)")
    print(f"    ({skipped} still reviewed:false, left in staging.)")
    print(f"    Provenance (timestamp + per-row sha256 + kaynak) appended to {manifest}")


def _row_hash(row):
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
def _read_jsonl(path):
    if not os.path.exists(path):
        sys.exit(f"ABORT: file not found: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                sys.exit(f"ABORT: {path} line {i} is not valid JSON.")
    return out


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Human-in-the-loop card authoring.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="write an example passages file")
    t.add_argument("--out", default="data/staging/passages.jsonl")
    t.set_defaults(func=cmd_template)

    b = sub.add_parser("build", help="passages -> draft rows (reviewed:false)")
    b.add_argument("--passages", required=True)
    b.add_argument("--out", default="data/staging/drafts.jsonl")
    b.set_defaults(func=cmd_build)

    p = sub.add_parser("promote", help="validate + append reviewed rows to train file")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", default="data/processed/task_sft.jsonl")
    p.add_argument("--reviewer", default=None, help="reviewer id recorded in the manifest")
    p.set_defaults(func=cmd_promote)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
