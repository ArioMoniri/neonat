#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mcq.py — synthetic Turkish neoperi MULTIPLE-CHOICE knowledge probe.
================================================================================
Teacher-generates single-answer MCQs from HELD-OUT passages (disjoint from
training AND the grounded benchmark), auto-QC-filters for passage-verifiability,
and labels everything synthetic/research. This is a KNOWLEDGE probe (separate from
the card behaviour benchmark) — a fair arena for a real medical model (MedGemma).

NOT clinician-validated. Correctness here means "verifiable from the passage",
which is sufficient for a research probe, not a clinical exam.

Usage:
  python build_mcq.py --passages data/corpus/passages.jsonl \
      --train data/processed/task_sft.synth.full.jsonl \
      --grounded data/benchmark/benchmark.jsonl \
      --teacher Qwen/Qwen2.5-72B-Instruct --n 100 --out data/benchmark/mcq.jsonl
  python build_mcq.py --selftest --out data/benchmark/mcq.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


TL = _load("train_lora", "train_lora.py")
SC = _load("synthesize_cards", "synthesize_cards.py")

MCQ_SYSTEM = (
    "Sen Türkçe tıbbi soru yazarısın. Verilen pasajdan, cevabı YALNIZCA pasajdan "
    "doğrulanabilen, tek doğru şıklı bir çoktan seçmeli soru üret. SADECE JSON üret: "
    "{\"soru\":\"...\",\"secenekler\":[\"...\",\"...\",\"...\",\"...\"],"
    "\"dogru\":\"A\",\"pasaj_dayanak\":\"<pasajdan birebir alıntı>\"}. "
    "Tam 4 şık ver; şıklara harf ekleme; 'dogru' A/B/C/D olsun; 3 çeldirici mantıklı "
    "ama yanlış olsun. Tanı/doz/tedavi kararı sorma; bilgi/tanım/eşik/risk sor."
)
LETTERS = ["A", "B", "C", "D"]


def used_ids(*paths):
    ids = set()
    for p in paths:
        if p and os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = o.get("provenance", {}).get("passage_id") or o.get("passage_id")
                if pid:
                    ids.add(pid)
    return ids


def qc(obj, passage):
    """Auto-QC (no clinician): passage-verifiability + well-formedness."""
    if not isinstance(obj, dict):
        return None
    opts = obj.get("secenekler")
    if not isinstance(opts, list) or len(opts) < 4:
        return None
    opts = [str(o).strip() for o in opts[:4] if str(o).strip()]
    if len(opts) != 4 or len({TL_norm(o) for o in opts}) != 4:   # 4 distinct
        return None
    dogru = str(obj.get("dogru", "")).strip().upper()[:1]
    if dogru not in LETTERS:
        return None
    dayanak = str(obj.get("pasaj_dayanak", "")).strip()
    if len(dayanak) < 8 or TL_norm(dayanak)[:40] not in TL_norm(passage):
        return None
    soru = str(obj.get("soru", "")).strip()
    if len(soru) < 8:
        return None
    return {"soru": soru, "secenekler": opts, "dogru": dogru, "pasaj_dayanak": dayanak}


def TL_norm(s):
    return (str(s).replace("İ", "i").replace("I", "ı").lower())


def main():
    ap = argparse.ArgumentParser(description="Build a synthetic TR neoperi MCQ probe.")
    ap.add_argument("--passages", default="data/corpus/passages.jsonl")
    ap.add_argument("--train", default="data/processed/task_sft.synth.full.jsonl")
    ap.add_argument("--grounded", default="data/benchmark/benchmark.jsonl")
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B")   # apache-2.0 (launcher passes MCQ_TEACHER)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/benchmark/mcq.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or args.dry_run:
        rows = [{
            "id": "mcq-0001", "soru": "Neonatal sarılıkta total serum bilirubin hangi "
            "iki değişkene göre yorumlanır?",
            "secenekler": ["Saat cinsinden yaş ve gebelik haftası", "Boy ve kilo",
                           "Anne yaşı ve kan grubu", "Doğum şekli ve mevsim"],
            "dogru": "A", "pasaj_dayanak": "age in hours and gestational age",
            "passage_id": "selftest-00001", "synthetic": True,
            "source": "teacher-generated", "reviewed": False}]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"==> [selftest] wrote {len(rows)} MCQ to {args.out}")
        return

    if not os.path.exists(args.passages):
        sys.exit(f"ABORT: {args.passages} not found (run build_corpus.py).")
    exclude = used_ids(args.train, args.grounded)
    heldout = []
    for line in open(args.passages, encoding="utf-8"):
        line = line.strip()
        if line:
            p = json.loads(line)
            if p.get("passage_id") not in exclude:
                heldout.append(p)
    if not heldout:
        sys.exit("ABORT: no held-out passages for MCQ (all used by train/grounded).")
    print(f"==> {len(heldout)} held-out passage(s) for MCQ (3-way disjoint)")

    model, tok = SC.load_teacher(args.teacher)
    dev = model.get_input_embeddings().weight.device
    import torch
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept, dropped = 0, 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, p in enumerate(heldout, 1):
            if kept >= args.n:
                break
            msgs = [{"role": "system", "content": MCQ_SYSTEM},
                    {"role": "user", "content": f"Pasaj:\n{p['passage']}"}]
            try:
                input_ids, attn = TL.ct_tensor(tok, msgs, device=dev,
                                               add_generation_prompt=True)
                kw = {} if attn is None else {"attention_mask": attn}
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=512, do_sample=True,
                                         temperature=0.7, top_p=0.9,
                                         eos_token_id=TL.response_terminator_id(tok),
                                         pad_token_id=(tok.pad_token_id or tok.eos_token_id), **kw)
                raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except Exception as e:  # noqa: BLE001
                print(f"    [{i}] gen error: {type(e).__name__}: {e!r}")
                dropped += 1
                continue
            mcq = qc(SC.extract_json(raw), p["passage"])
            if not mcq:
                dropped += 1
                continue
            mcq.update({"id": f"mcq-{kept + 1:04d}", "passage_id": p.get("passage_id"),
                        "synthetic": True, "source": "teacher-generated", "reviewed": False})
            fh.write(json.dumps(mcq, ensure_ascii=False) + "\n")
            kept += 1
            if i % 25 == 0:
                print(f"    [{i}] kept={kept} dropped={dropped}")
    if kept == 0:
        sys.exit("ABORT: no valid MCQ produced (QC too strict or teacher weak).")
    print(f"==> Wrote {kept} SYNTHETIC MCQ to {args.out} (dropped {dropped}). "
          "Research probe — teacher-generated, NOT clinician-validated.")


if __name__ == "__main__":
    main()
