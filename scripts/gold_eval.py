#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gold_eval.py — VALIDATE THE RULER. Correlate the reference-free benchmark metrics
against clinician ratings, so a leaderboard number means something.
================================================================================
Both the engineering and clinical reviews said the same thing: the benchmark is
steered by a metric of unknown accuracy on synthetic data. Until those metrics are
shown to track human judgment, every ranking is plumbing-validation only.

Workflow:
  1. python gold_eval.py template --out data/gold/gold.jsonl        # blank rating sheet
  2. >=2 clinicians independently score each card (0..2) on:
        grounding_h, safety_h, acuity_h, usefulness_h                # human scores
  3. python gold_eval.py analyze --gold data/gold/gold.jsonl
        -> inter-rater agreement (Cohen's/Fleiss κ) + correlation of each
           reference-free metric (from benchmark score_case) vs the human means.

Gold row schema (one JSON per line):
  {"id","system","user","card":"<model card JSON string>",
   "ratings":[{"rater":"dr_a","grounding_h":2,"safety_h":2,"acuity_h":1,"usefulness_h":2},
              {"rater":"dr_b",...}]}
This is the blocking dependency before trusting any model ranking (ROADMAP challenge #1).
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ("grounding_h", "safety_h", "acuity_h", "usefulness_h")


def _tl():
    s = importlib.util.spec_from_file_location("train_lora", os.path.join(_HERE, "train_lora.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def cmd_template(args):
    rows = [{
        "id": "gold-0001",
        "system": _tl().GUARDRAIL_SYSTEM,
        "user": "Hasta bağlamı: [gerçek/temsili olgu]. Kılavuz pasajı: [pasaj].",
        "card": "{\"onerilen_sorular\":[],\"onerilen_tetkikler\":[],\"eksik_veriler\":[],"
                "\"kaynak\":\"\",\"uyari\":\"\",\"kirmizi_bayraklar\":[]}",
        "ratings": [{"rater": "dr_a", **{a: None for a in AXES}},
                    {"rater": "dr_b", **{a: None for a in AXES}}],
    }]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"==> Wrote rating template to {args.out}. Each rater fills 0..2 per axis "
          f"({', '.join(AXES)}); populate 'card' with real model outputs.")


def _corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    sx = sum(p[0] for p in pairs); sy = sum(p[1] for p in pairs)
    mx, my = sx / n, sy / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    vx = sum((p[0] - mx) ** 2 for p in pairs); vy = sum((p[1] - my) ** 2 for p in pairs)
    if vx == 0 or vy == 0:
        return 0.0
    return round(cov / (vx * vy) ** 0.5, 3)


def _fleiss_kappa(items):
    """items: list of {category: count}; simple Fleiss κ over a fixed category set."""
    if not items:
        return None
    cats = sorted({c for it in items for c in it})
    N = len(items)
    n = sum(items[0].values())
    if n < 2:
        return None
    p = {c: sum(it.get(c, 0) for it in items) / (N * n) for c in cats}
    Pbar = sum((sum(it.get(c, 0) ** 2 for c in cats) - n) / (n * (n - 1)) for it in items) / N
    Pe = sum(v * v for v in p.values())
    return round((Pbar - Pe) / (1 - Pe), 3) if (1 - Pe) else None


def cmd_analyze(args):
    TL = _tl()
    # import benchmark scorer (reference-free metrics)
    sb = importlib.util.spec_from_file_location("bm", os.path.join(_HERE, "benchmark.py"))
    bm = importlib.util.module_from_spec(sb)
    sb.loader.exec_module(bm)

    rows = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    rated = [r for r in rows if any(
        rt.get(a) is not None for rt in r.get("ratings", []) for a in AXES)]
    if len(rated) < 3:
        sys.exit(f"ABORT: only {len(rated)} rated rows — need clinician ratings first "
                 "(fill the template, >=2 raters).")

    # Inter-rater agreement per axis (κ over 0/1/2 categories).
    print("== Inter-rater agreement (Fleiss κ, 0..2) ==")
    for a in AXES:
        items = []
        for r in rated:
            vals = [rt.get(a) for rt in r["ratings"] if rt.get(a) is not None]
            if len(vals) >= 2:
                items.append({c: vals.count(c) for c in (0, 1, 2)})
        k = _fleiss_kappa(items)
        print(f"  {a:14s} κ={k}  (n={len(items)})")

    # Human mean per axis + reference-free metric per card.
    human = {a: [] for a in AXES}
    metric = {}
    for r in rated:
        for a in AXES:
            vals = [rt.get(a) for rt in r["ratings"] if rt.get(a) is not None]
            human[a].append(sum(vals) / len(vals) if vals else None)
        sc = bm.score_case({"category": "grounded", "user": r.get("user", "")}, r.get("card", ""))
        for mk in ("format", "safety", "grounding", "tr_purity", "acuity", "helpful"):
            metric.setdefault(mk, []).append(sc.get(mk))

    print("\n== Correlation: reference-free metric vs clinician axis (Pearson r) ==")
    print(f"  {'metric':12s} " + " ".join(f"{a[:-2]:>10s}" for a in AXES))
    for mk, xs in metric.items():
        cells = " ".join(f"{str(_corr(xs, human[a])):>10s}" for a in AXES)
        print(f"  {mk:12s} {cells}")
    print("\nNOTE: |r| >= ~0.4 suggests the automated metric tracks that clinical axis. "
          "Low/again-zero correlation means the leaderboard number is NOT measuring it — "
          "freeze that metric as plumbing-only until the gold set is larger.")


def main():
    ap = argparse.ArgumentParser(description="Validate benchmark metrics vs clinician ratings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template"); t.add_argument("--out", default="data/gold/gold.jsonl")
    t.set_defaults(func=cmd_template)
    a = sub.add_parser("analyze"); a.add_argument("--gold", default="data/gold/gold.jsonl")
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
