#!/usr/bin/env python3
"""compare_outputs.py — show, side by side, what fine-tuning CHANGED.

For one registry student, load the raw BASE model and the BASE+adapter, generate
a suggestion-card on a handful of benchmark cases spanning categories, and print
both outputs so you can eyeball the behavior change:
  - base prescribes / gives imperative orders   -> ft returns a grounded card
  - base invents a `kaynak` on an empty passage -> ft refuses / flags missing data
  - base misses the sick baby                   -> ft escalates (kirmizi_bayraklar)

Reference-free SCORING of these axes lives in leaderboard.md; this tool is for
reading the raw text, not grading.

Usage (on the server, inside the venv):
  python scripts/compare_outputs.py --model kumru
  python scripts/compare_outputs.py --model qwen3-4b --n 6
  python scripts/compare_outputs.py --base vngrs-ai/Kumru-2B \
      --adapter models/kumru-neoperi-synth --n 5
"""
import argparse
import gc
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load("evaluate", "evaluate.py")   # reuse the exact generate()/load_model() path
TL = EV.TL


def resolve_from_registry(model_name):
    conf = os.path.join(_HERE, "..", "config", "models.conf")
    with open(conf, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0] == model_name:
                return parts[1]
    sys.exit(f"ABORT: model '{model_name}' not in config/models.conf")


def load_base_only(base_id):
    """Raw base with its OWN tokenizer (parity with how benchmark.py scores bases)."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, device_map="auto", **TL.hf_dtype_kwargs())
    m.eval()
    return m, tok


def pick_cases(bench_path, n):
    rows = [json.loads(l) for l in open(bench_path, encoding="utf-8") if l.strip()]
    chosen, per_cat = [], {}
    # one of each category first (safety-critical ones surface), then fill in order
    for r in rows:
        c = r.get("category", "grounded")
        if per_cat.get(c, 0) < 1:
            chosen.append(r)
            per_cat[c] = per_cat.get(c, 0) + 1
        if len(chosen) >= n:
            return chosen
    for r in rows:
        if r not in chosen:
            chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen


def _free(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Base vs fine-tuned card outputs, side by side.")
    ap.add_argument("--model", help="registry name in config/models.conf (e.g. kumru)")
    ap.add_argument("--base", help="base HF id (overrides registry lookup)")
    ap.add_argument("--adapter", help="adapter dir (default models/<model>-neoperi-synth)")
    ap.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--maxchars", type=int, default=900)
    args = ap.parse_args()

    base_id = args.base or (resolve_from_registry(args.model) if args.model else None)
    if not base_id:
        sys.exit("ABORT: pass --model NAME or --base HF_ID")
    adapter = args.adapter or (f"models/{args.model}-neoperi-synth" if args.model else None)
    if not adapter or not os.path.isdir(adapter):
        sys.exit(f"ABORT: adapter dir not found: {adapter}")
    if not os.path.exists(args.benchmark):
        sys.exit(f"ABORT: benchmark not found: {args.benchmark} (run scripts/run_benchmark.sh first)")

    cases = pick_cases(args.benchmark, args.n)
    print(f"### compare  base={base_id}  adapter={adapter}  cases={len(cases)}")
    cats = {}
    for c in cases:
        cats[c.get("category", "?")] = cats.get(c.get("category", "?"), 0) + 1
    print(f"### categories: {cats}\n")

    # Load one model at a time to keep VRAM low.
    print("==> loading BASE (adapter OFF) ...", flush=True)
    bmodel, btok = load_base_only(base_id)
    base_out = [EV.generate(bmodel, btok, c["system"], c["user"]) for c in cases]
    _free(bmodel)

    print("==> loading BASE+ADAPTER (fine-tuned) ...", flush=True)
    fmodel, ftok = EV.load_model(base_id, adapter)
    ft_out = [EV.generate(fmodel, ftok, c["system"], c["user"]) for c in cases]
    _free(fmodel)

    for c, b, f in zip(cases, base_out, ft_out):
        print("\n" + "=" * 78)
        print(f"[{c.get('category', '?')}]  id={c.get('id', '?')}")
        u = " ".join(str(c.get("user", "")).split())
        print(f"USER: {u[:360]}")
        print("-" * 78)
        print("BASE >>>")
        print(b.strip()[:args.maxchars] or "(empty)")
        print("-" * 78)
        print("FT   >>>")
        print(f.strip()[:args.maxchars] or "(empty)")
    print("\n(These are raw generations. Scored axes: leaderboard.md.)")


if __name__ == "__main__":
    main()
