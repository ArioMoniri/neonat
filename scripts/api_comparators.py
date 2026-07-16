#!/usr/bin/env python3
"""api_comparators.py — generate suggestion-cards for the benchmark cases with FRONTIER
CLOSED models (Anthropic / OpenAI / Google) so they can be scored against our fine-tuned
students. Writes {label, id, output} JSONL that benchmark.py folds via --precomputed.

Keys + exact model ids come from ENV (never hard-code a possibly-hallucinated id; the
NEJM protocol requires logging the exact id + access date). A provider is skipped if its
key or SDK is absent. This does inference only — no financial or account actions.

  ANTHROPIC_API_KEY  ANTHROPIC_MODEL (default claude-opus-4-8)
  OPENAI_API_KEY     OPENAI_MODEL    (default gpt-5)
  GOOGLE_API_KEY     GOOGLE_MODEL    (default gemini-2.5-pro)

Usage:
  python scripts/api_comparators.py --benchmark data/benchmark/benchmark.jsonl \
      --out data/benchmark/api_outputs.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TL = _load("train_lora", "train_lora.py")
GUARDRAIL = TL.GUARDRAIL_SYSTEM


# --- provider adapters: return generated text, or raise ----------------------
def gen_anthropic(system, user, model):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=800, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(getattr(b, "text", "") for b in msg.content)


def gen_openai(system, user, model):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model, max_tokens=800,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def gen_google(system, user, model):
    from google import genai
    client = genai.Client()
    r = client.models.generate_content(model=model, contents=[system + "\n\n" + user])
    return r.text or ""


# --- auto-discovery: ask each provider what it OFFERS, pick the strongest, log it -----
# Preference tiers (best -> worst); newest wins within a tier (reverse lexical sort). This
# stays current without hard-coding an id that may 404, and logs the exact id for the record.
_PREF = {
    "anthropic": [r"opus-4", r"opus", r"sonnet-4", r"sonnet"],
    "openai":    [r"gpt-5\.\d", r"gpt-5", r"o[45]", r"o3", r"gpt-4\.\d", r"gpt-4o"],
    "google":    [r"gemini-3.*pro", r"gemini-2\.5-pro", r"gemini.*pro", r"gemini.*flash"],
}
_BAD = __import__("re").compile(
    r"embed|tts|whisper|image|dall|realtime|audio|moderation|search|transcribe|vision|tuning|nano",
    __import__("re").IGNORECASE)


def list_model_ids(provider):
    try:
        if provider == "anthropic":
            import anthropic
            return [m.id for m in anthropic.Anthropic().models.list().data]
        if provider == "openai":
            from openai import OpenAI
            return [m.id for m in OpenAI().models.list().data]
        if provider == "google":
            from google import genai
            return [m.name.split("/")[-1] for m in genai.Client().models.list()]
    except Exception as e:  # noqa: BLE001
        print(f"    [{provider}] models.list failed ({type(e).__name__}: {e}); using default.")
    return []


def pick_best(provider, ids):
    """Within the best matching preference tier, prefer the FLAGSHIP (not mini/lite) and
    the newest version. Returns the chosen id, or None."""
    import re
    light = re.compile(r"mini|lite|nano|small|flash-8b|haiku", re.IGNORECASE)

    def rank(i):
        nums = [int(n) for n in re.findall(r"\d+", i)][:4]
        nums += [0] * (4 - len(nums))
        return (1 if light.search(i) else 0,) + tuple(-n for n in nums)

    cand = [i for i in ids if not _BAD.search(i)]
    for pat in _PREF.get(provider, []):
        hits = [i for i in cand if re.search(pat, i, re.IGNORECASE)]
        if hits:
            return sorted(hits, key=rank)[0]
    return sorted(cand, key=rank)[0] if cand else None


PROVIDERS = [
    ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-opus-4-8",  gen_anthropic),
    ("openai",    "OPENAI_API_KEY",    "OPENAI_MODEL",    "gpt-5",            gen_openai),
    ("google",    "GOOGLE_API_KEY",    "GOOGLE_MODEL",    "gemini-2.5-pro",   gen_google),
]


def main():
    ap = argparse.ArgumentParser(description="Frontier API comparator card generation.")
    ap.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    ap.add_argument("--out", default="data/benchmark/api_outputs.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap #cases (0 = all)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between API calls")
    args = ap.parse_args()

    if not os.path.exists(args.benchmark):
        sys.exit(f"ABORT: benchmark not found: {args.benchmark} (run build_benchmark.py first)")
    cases = [json.loads(l) for l in open(args.benchmark, encoding="utf-8") if l.strip()]
    if args.limit:
        cases = cases[:args.limit]

    active = []
    for name, key_env, model_env, default_model, fn in PROVIDERS:
        if not os.environ.get(key_env):
            print(f"==> {name}: {key_env} not set — skipping.")
            continue
        pinned = os.environ.get(model_env)
        if pinned:
            model = pinned
            print(f"==> {name}: pinned model='{model}'")
        else:
            model = pick_best(name, list_model_ids(name)) or default_model
            print(f"==> {name}: auto-selected strongest available model='{model}' "
                  f"(set {model_env} to pin a specific id)")
        active.append((name, model, fn))
    if not active:
        print("==> No frontier API keys set — nothing to generate. (Open comparators still run.)")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for name, model, fn in active:
            label = f"{name}:{model}"
            ok = 0
            for c in cases:
                system = c.get("system", GUARDRAIL)
                user = c.get("user", "")
                try:
                    out = fn(system, user, model)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"    [{label}] {c.get('id')}: {type(e).__name__}: {e!r}")
                    out = ""
                fh.write(json.dumps({"label": label, "id": c.get("id"),
                                     "output": out}, ensure_ascii=False) + "\n")
                written += 1
                time.sleep(args.pause)
            print(f"==> {label}: generated {ok}/{len(cases)} cards")
    print(f"==> Wrote {written} rows to {args.out}. Fold with: "
          f"benchmark.py --precomputed {args.out}")


if __name__ == "__main__":
    main()
