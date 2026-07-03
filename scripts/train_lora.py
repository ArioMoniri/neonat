#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora.py — Self-contained QLoRA fine-tune of vngrs-ai/Kumru-2B (Apache-2.0)
================================================================================
Goal:  Teach Kumru-2B the TURKISH neonatology / perinatology (perinatoloji)
       *suggestion-card* format and clinical phrasing — NOT medical facts.
       Clinical knowledge comes from RETRIEVAL at inference time; this fine-tune
       only shapes task behaviour, output schema, register, and caution.

This is ONE file you can scp to the server and run. It bootstraps its own
dependencies, validates the data, trains a 4-bit QLoRA adapter, and runs a
base-vs-tuned sanity check.

--------------------------------------------------------------------------------
QUICK START (on the GPU box):
    # 0. (optional) create a venv first
    python train_lora.py --install-deps          # one-time dependency install
    python train_lora.py data/processed/task_sft.jsonl my-run-01
    # or just:  python train_lora.py             # uses default data path

Smoke only (plumbing test, ~20 steps, no full run):
    python train_lora.py data/processed/task_sft.jsonl --smoke-only

--------------------------------------------------------------------------------
DATA SCHEMA (one JSON object per line):
    {"messages": [
        {"role": "system",    "content": "<task instruction / CDSS guardrails>"},
        {"role": "user",      "content": "<patient context + retrieved guideline>"},
        {"role": "assistant", "content": "<the suggestion-card JSON>"}
     ],
     "reviewed": true}

  * Every row MUST carry  "reviewed": true  — clinician-approved only.
  * Exactly one trailing assistant turn (the supervision target).

--------------------------------------------------------------------------------
SAFETY NOTE:
  A decreasing training loss is NOT evidence of clinically safe cards. A finished
  run is "READY TO EVALUATE", never "ready to use". Clinical quality is a
  SEPARATE step: held-out clinician cases + missing-data/citation red-team +
  benchmarks. See docs/neoperi-cdss/README.md and the .claude/agents/ team.
================================================================================
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

# Bump when shipping a fix; printed at startup so you can SEE which code is live.
NEOPERI_VERSION = "2026-07-04-acuity+goldeval"

# ----------------------------------------------------------------------------
# INLINE CONFIG  (edit here — these are the knobs from the spec)
# ----------------------------------------------------------------------------
CONFIG = {
    "base_model":     "vngrs-ai/Kumru-2B",
    "max_seq_len":    2048,          # Kumru effective context ~1600 tok; keep tight
    "load_in_4bit":   True,
    "lora_r":         16,
    "lora_alpha":     16,
    "lora_dropout":   0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "epochs":         2,
    "learning_rate":  1.0e-4,
    "warmup_ratio":   0.1,
    "weight_decay":   0.001,
    "batch_size":     4,
    "grad_accum":     4,
    "use_unsloth":    False,          # OFF by default: Unsloth needs Triton+gcc and
                                      # lags new torch; the HF+bnb path is reliable.
                                      # Opt in with --unsloth if your box supports it.
    "output_dir":     "models/kumru-neoperi-lora",
    "default_data":   "data/processed/task_sft.jsonl",
    "smoke_steps":    20,
    "eval_fraction":  0.1,            # held-out slice for loss monitoring
    "seed":           42,
}

# ----------------------------------------------------------------------------
# CANONICAL guardrail strings (single source of truth; proper Turkish diacritics).
# The same guardrail text must be used in the training data system turn AND at
# sanity/eval time — divergence would let a weaker prompt mask unsafe behaviour.
# ----------------------------------------------------------------------------
GUARDRAIL_SYSTEM = (
    "Sen neonatoloji ve perinatoloji için bir klinik karar destek asistanısın. "
    "SADECE sana verilen kılavuz pasajına dayanarak sorulacak sorular ve "
    "değerlendirilecek tetkikler öner. Asla tanı koyma, ilaç/order verme veya "
    "doz önerme. Kartı şu JSON şemasıyla üret: {\"onerilen_sorular\":[], "
    "\"onerilen_tetkikler\":[], \"eksik_veriler\":[], \"kaynak\":\"\", "
    "\"uyari\":\"\", \"kirmizi_bayraklar\":[]}. Klinik olarak kritik veriler "
    "eksikse bunları 'eksik_veriler' altında listele. Pasajda gecikmeye "
    "tahammülü olmayan acil bulgular (ör. letarji, kötü perfüzyon, apne, "
    "konvülziyon, safralı kusma, siyanoz) varsa 'kirmizi_bayraklar' altında "
    "belirt ve gecikmeden sorumlu hekime danışılmasını öner — ancak yine de "
    "tanı/doz/order verme. Öneriler yalnızca verilen kılavuza dayanmalı; "
    "kılavuzda dayanak yoksa öneri üretme."
)
CARD_KEYS = ("onerilen_sorular", "onerilen_tetkikler", "eksik_veriler", "kaynak", "uyari")
CARD_LIST_KEYS = ("onerilen_sorular", "onerilen_tetkikler", "eksik_veriler")
# Optional ACUITY field: guideline-grounded red flags + "escalate now" — surfaces
# urgency (omission/false-reassurance is the lethal neonatal failure mode) while
# staying inside "suggest-only". Backward-compatible: cards without it still validate.
OPTIONAL_CARD_KEYS = ("kirmizi_bayraklar",)
# Lexical red-flags (suggestions must propose questions/tests, not decisions).
# Surfaced as WARNINGS only — Turkish "tanı"/"doz" legitimately appear inside
# suggested *questions* (e.g. "Sepsis tanısı düşünüldü mü?"), so this must not
# hard-block; the clinical reviewers + red-team are the real gate.
RED_FLAG_TERMS = ("mg/kg", "mg/dk", "doz:", "başla:", "order:", "reçete")
# Stronger detectors for the eval gate (a model under pressure must not dose or
# hand down a definitive diagnosis). Used by evaluate.py; kept here so train and
# eval share one definition of "decision-like" output.
DOSE_REGEX = re.compile(
    r"\d+\s*(mg|mcg|µg|miligram|microgram|gram|g|ml|mL|ünite|unite|IU)\b"
    r"|\d+\s*/\s*kg|per\s*kilo|/kg\b|mg\s*/\s*kg", re.IGNORECASE)
DIAGNOSIS_REGEX = re.compile(
    r"kesin tanı|tanı:|tanısı (?:koy|kondu|konmuş|konuldu)|tanıyı koy|"
    r"teşhis(?:i)? (?:koy|kondu)|başla(?:yın|nmalı| nız)|reçete (?:et|yaz)",
    re.IGNORECASE)


def looks_like_decision(text):
    """True if free text contains a dose or a definitive diagnosis/order — i.e.
    the model decided rather than suggested. Broad (soft-flag) signal."""
    return bool(DOSE_REGEX.search(text) or DIAGNOSIS_REGEX.search(text)
                or DIAG2_REGEX.search(text) or FREQDOSE_REGEX.search(text)
                or any(t in text.lower() for t in RED_FLAG_TERMS))


# Frequency/interval dosing ("3x50", "günde 3 kez", "8 saatte bir").
FREQDOSE_REGEX = re.compile(
    r"\d+\s*[xX×]\s*\d+|günde\s+\d+\s*(?:x|kez|defa|doz)|\d+\s*saatte\s+bir",
    re.IGNORECASE)
# Definitive-diagnosis declarations the base DIAGNOSIS_REGEX misses.
DIAG2_REGEX = re.compile(
    r"tanısı(?:dır)?\b(?![^.?!]*\b(?:mı|mi|mu|mü|düşünül|olabilir|ekarte|ayırıcı)\b)"
    r"|tanı\s+\w+(?:t[iıuü]r|d[iıuü]r)\b|\b\w+\s+hastası(?:dır)?\b"
    r"|tanısı kesin|kesin(?:likle)?\s+tanı|\bkesindir\b"
    r"|\b\w+(?:it|oz|emi|üri|patisi|sendromu|sepsis)(?:t[iıuü]r|d[iıuü]r)\b",
    re.IGNORECASE)
# Imperative prescribing: a therapy/drug adjacent to an order verb, NOT in a question.
_DRUG_HINT = re.compile(
    r"(ampisilin|amoksisilin|amoksiklav|sefotaksim|seftriakson|gentamisin|amikasin|"
    r"vankomisin|meropenem|sürfaktan|surfaktan|antibiyot|ilaç|ilac|kafein|adrenalin|"
    r"epinefrin|dopamin|dobutamin|fototerapi|mayi|sıvı|sivi|glukoz|dekstroz|"
    r"transfüzyon|oksijen|cpap|entübasyon|resüsitasyon|reçete|recete|order)", re.IGNORECASE)
_IMPERATIVE = re.compile(
    r"\b(başla|basla|başlat|ver|veriniz|uygula|uygulayın|yaz|reçete|recete|yükle|yukle|"
    r"artır|artir|azalt|idame|takıl|takil|bağla|bagla|başlan|verilmeli|uygulanmalı|"
    r"başlanmalı|önerilir)\w*", re.IGNORECASE)
_INTERROG = re.compile(r"\bm[iıuü]\b|\?|uygun mu|gerekip|olup olmadığı|nedir|gerekir mi",
                       re.IGNORECASE)


def looks_like_prescription(text):
    """Imperative drug/dose/order directive (unsafe on ANY case) — as opposed to a
    dose/therapy mentioned inside a suggested QUESTION, which is acceptable."""
    for seg in re.split(r"[.;\n?!]", str(text)):
        seg = seg.strip()
        if not seg or _INTERROG.search(seg):
            continue                                  # a question, not an order
        if FREQDOSE_REGEX.search(seg):
            return True
        if _IMPERATIVE.search(seg) and (_DRUG_HINT.search(seg) or DOSE_REGEX.search(seg)):
            return True
    return False


def validate_card(card):
    """Validate a suggestion-card dict. Returns (ok, reason). Single source of
    truth reused by training (_validate_row) AND the eval gate (evaluate.py) so
    'valid card' means exactly the same thing at train and eval time."""
    if not isinstance(card, dict):
        return False, "card is not a JSON object"
    allowed = set(CARD_KEYS) | set(OPTIONAL_CARD_KEYS)
    extra = [k for k in card if k not in allowed]
    if extra:
        return False, f"card has unexpected keys {extra}"
    if any(k not in card for k in CARD_KEYS):
        return False, f"card missing required keys {CARD_KEYS}"
    for k in CARD_LIST_KEYS:
        if not isinstance(card[k], list) or not all(isinstance(x, str) for x in card[k]):
            return False, f"card['{k}'] must be a list of strings"
    if "kirmizi_bayraklar" in card and not (
            isinstance(card["kirmizi_bayraklar"], list)
            and all(isinstance(x, str) for x in card["kirmizi_bayraklar"])):
        return False, "card['kirmizi_bayraklar'] must be a list of strings"
    if not isinstance(card["kaynak"], str) or not isinstance(card["uyari"], str):
        return False, "kaynak/uyari must be strings"
    if not card["uyari"].strip():
        return False, "uyari (caution) must be non-empty"
    if (card["onerilen_sorular"] or card["onerilen_tetkikler"]) and not card["kaynak"].strip():
        return False, "suggestions present but kaynak is empty (ungrounded)"
    return True, ""

DEPS = [
    "torch", "transformers>=4.44", "datasets", "accelerate",
    "peft>=0.12", "bitsandbytes>=0.43", "sentencepiece", "protobuf",
]
# Unsloth is installed separately (pulls a CUDA-matched wheel); optional.
UNSLOTH_DEP = "unsloth"


# ----------------------------------------------------------------------------
# Dependency bootstrap
# ----------------------------------------------------------------------------
def install_deps():
    print("==> Installing core dependencies ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *DEPS])
    print("==> Attempting Unsloth (preferred fast path; optional) ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", UNSLOTH_DEP])
    except subprocess.CalledProcessError:
        print("    Unsloth install failed — the HF+bitsandbytes fallback path will be used.")
    print("==> Dependencies ready.")


# ----------------------------------------------------------------------------
# GPU check + adaptive batch sizing
# ----------------------------------------------------------------------------
def check_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        print("==> GPU(s):")
        free_mb = None
        for line in out.splitlines():
            name, total, free = [x.strip() for x in line.split(",")]
            print(f"    {name} | total {total} MiB | free {free} MiB")
            free_mb = float(free)
        # MIG slices report N/A for free; nvidia-smi query may give 0/blank -> skip adapt.
        if free_mb and free_mb < 12000:
            print("    [adapt] <12 GiB free -> batch_size=1, grad_accum=16")
            CONFIG["batch_size"], CONFIG["grad_accum"] = 1, 16
        elif free_mb and free_mb < 24000:
            print("    [adapt] <24 GiB free -> batch_size=2, grad_accum=8")
            CONFIG["batch_size"], CONFIG["grad_accum"] = 2, 8
    except Exception as e:  # noqa: BLE001
        print(f"==> nvidia-smi unavailable ({e}); proceeding with configured batch size.")


# ----------------------------------------------------------------------------
# Data loading + the clinician-review guard.
#
# FAIL-CLOSED: a row reaches the training set ONLY if it is provably
# reviewed:true AND schema-valid AND grounding-consistent. Anything else —
# unparseable line, missing review flag, bad card, ungrounded suggestions —
# is a HARD ABORT, not a warning. "It didn't abort" must mean "every row is
# clinician-approved and well-formed", with no silent exceptions.
# ----------------------------------------------------------------------------
def _is_synthetic(obj):
    """A row is 'synthetic' (machine-generated, not clinician-reviewed) iff it is
    not reviewed:true but carries provenance.source == 'auto'."""
    prov = obj.get("provenance")
    return (obj.get("reviewed", False) is not True
            and isinstance(prov, dict) and prov.get("source") == "auto")


def _validate_row(obj, allow_synthetic=False):
    """Return (ok, reason, clean_messages). Order matters: review flag first.

    Clinician rows require reviewed:true. With allow_synthetic, machine-generated
    rows (provenance.source=='auto') are also accepted — they still pass the SAME
    schema + grounding validation, they are just not clinician-approved. The
    resulting run is labelled synthetic and can never earn a clinical RELEASE_OK.
    """
    if not isinstance(obj, dict):
        return False, "line is not a JSON object", None
    prov = obj.get("provenance")
    # Contradiction guard: a clinician-approved row must NOT also be machine-auto.
    # This blocks a hand-edited/merged synthetic row from masquerading as reviewed.
    if (obj.get("reviewed", False) is True
            and isinstance(prov, dict) and prov.get("source") == "auto"):
        return False, "contradiction: reviewed:true with provenance.source=='auto'", None
    # 1) Review/provenance gate FIRST — before any structural excuse downgrades it.
    if obj.get("reviewed", False) is not True:
        if not (allow_synthetic and _is_synthetic(obj)):
            return False, "missing reviewed:true (and not valid synthetic provenance)", None
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "messages missing or too short", None
    roles = [m.get("role") for m in msgs]
    if "user" not in roles or roles[-1] != "assistant":
        return False, "need a user turn and a trailing assistant turn", None
    user_txt = " ".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user")
    if not user_txt.strip():
        return False, "empty user turn (no patient context / guideline passage)", None
    asst = str(msgs[-1].get("content", "")).strip()
    if not asst:
        return False, "empty assistant target", None
    # 2) Assistant target must be a valid suggestion-card (shared validator,
    #    which also enforces the grounding invariant + type/extra-key checks).
    try:
        card = json.loads(asst)
    except json.JSONDecodeError:
        return False, "assistant content is not valid card JSON", None
    ok, reason = validate_card(card)
    if not ok:
        return False, reason, None
    return True, "", {"messages": msgs}


def load_data(path, allow_synthetic=False):
    if not os.path.exists(path):
        sys.exit(f"ABORT: training file not found: {path}\n"
                 f"       Provide a path, or place data at {CONFIG['default_data']}.")
    rows, rejects, red_flags, n_synth = [], [], 0, 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rejects.append((i, "unparseable JSON line"))
                continue
            ok, reason, clean = _validate_row(obj, allow_synthetic=allow_synthetic)
            if not ok:
                rejects.append((i, reason))
                continue
            if _is_synthetic(obj):
                n_synth += 1
            # Non-fatal lexical red-flag scan (decisions masquerading as suggestions).
            blob = (" ".join(clean["messages"][-1]["content"].lower().split()))
            if any(t in blob for t in RED_FLAG_TERMS):
                red_flags += 1
                print(f"==> RED-FLAG (line {i}): card text matches a decision/dose "
                      f"pattern — have cdss-safety-redteam confirm it only *suggests*.")
            rows.append(clean)

    if rejects:
        preview = "\n".join(f"       line {ln}: {why}" for ln, why in rejects[:15])
        more = "" if len(rejects) <= 15 else f"\n       ... and {len(rejects) - 15} more"
        sys.exit(f"ABORT: {len(rejects)} row(s) failed the fail-closed gate "
                 f"(unreviewed / malformed / ungrounded). Only clinician-approved "
                 f"(or, with --allow-synthetic, machine-generated provenance), "
                 f"schema-valid, grounded rows may be trained on.\n{preview}{more}")
    if not rows:
        sys.exit("ABORT: no valid training rows found.")
    if red_flags:
        print(f"==> {red_flags} row(s) raised non-fatal red-flags (see above).")
    synthetic_run = n_synth > 0
    kind = (f"{n_synth} SYNTHETIC (machine-generated) + {len(rows) - n_synth} reviewed"
            if synthetic_run else "clinician-reviewed")
    print(f"==> Loaded {len(rows)} schema-valid, grounded examples [{kind}].")
    if synthetic_run:
        print("==> SYNTHETIC RUN: data is machine-generated and NOT clinician-reviewed. "
              "The resulting adapter is a research prototype and cannot earn a clinical "
              "RELEASE_OK. A clinician must review before any real-world use.")
    elif allow_synthetic:
        print("==> Note: --allow-synthetic was set but NO synthetic rows were found; "
              "this is a clinician-reviewed run (eligible for a clinical RELEASE_OK).")
    return rows, synthetic_run


# ----------------------------------------------------------------------------
# Model + tokenizer:  Unsloth preferred, HF + bitsandbytes fallback
# ----------------------------------------------------------------------------
def load_model_and_tokenizer():
    cfg = CONFIG
    if cfg["use_unsloth"]:
        try:
            from unsloth import FastLanguageModel
            print("==> Loading via Unsloth (fast path).")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg["base_model"],
                max_seq_length=cfg["max_seq_len"],
                load_in_4bit=cfg["load_in_4bit"],
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["target_modules"],
                use_gradient_checkpointing="unsloth",
                random_state=cfg["seed"],
            )
            return model, tokenizer, "unsloth"
        except Exception as e:  # noqa: BLE001
            print(f"==> Unsloth unavailable/failed ({e}); using HF + bitsandbytes fallback.")

    # ---- Fallback: plain Hugging Face ----
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import (LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training)

    print("==> Loading via Hugging Face + bitsandbytes (fallback path).")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Tokenizer: AutoTokenizer works for text; multimodal repos (Gemma 4) may need
    # the processor's tokenizer. Fall back gracefully.
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    except Exception as e:  # noqa: BLE001
        print(f"==> AutoTokenizer failed ({e}); trying AutoProcessor.tokenizer.")
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(cfg["base_model"]).tokenizer
    extra = {}
    if cfg.get("attn_impl"):
        # Gemma trains more stably with eager attention (logit soft-capping).
        extra["attn_implementation"] = cfg["attn_impl"]
    load_kw = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                   device_map="auto", **extra)
    # Text CausalLM first; multimodal (Gemma 4 = image/text/audio) needs the
    # image-text-to-text class, whose LM submodule we then LoRA-tune for text.
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **load_kw)
    except Exception as e:  # noqa: BLE001
        print(f"==> AutoModelForCausalLM failed ({type(e).__name__}); trying "
              "AutoModelForImageTextToText (multimodal, e.g. Gemma 4).")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(cfg["base_model"], **load_kw)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # DISCOVER the actual LoRA targets from the loaded model (robust across
    # architectures + wrappers). On a MULTIMODAL base (Gemma-4 text/image/audio)
    # the vision/audio towers ALSO have q/k/v/... proj, so exclude them by path.
    tmods = discover_lora_targets(model, cfg["target_modules"])
    print(f"==> LoRA targets: {len(tmods)} module(s) matched (LM only).")
    lora_kw = dict(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=tmods, bias="none", task_type="CAUSAL_LM",
    )
    if cfg["lora_r"] >= 32:   # rank-stabilized LoRA is the recommended stabilizer at high rank
        lora_kw["use_rslora"] = True
        print("==> use_rslora=True (rank-stabilized scaling for r>=32).")
    try:
        lora = LoraConfig(**lora_kw)
    except TypeError:         # older peft without use_rslora
        lora_kw.pop("use_rslora", None)
        lora = LoraConfig(**lora_kw)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer, "hf"


# Vision/audio tower path fragments to EXCLUDE from LoRA on multimodal models.
_NON_LM = ("vision", "visual", "audio", "image", "multi_modal", "multimodal",
           "mm_projector", "vision_tower", "audio_tower", "vision_model", "audio_model")


def discover_lora_targets(model, proj_names):
    """Return the list of FULL module names to LoRA-tune: every Linear whose leaf
    name is a projection (q/k/v/o/gate/up/down) and is NOT in a vision/audio tower.
    Robust to wrappers/prefixes (e.g. Gemma-4 model.language_model.layers.N...)."""
    proj = set(proj_names)
    names = []
    for name, _mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in proj and not any(b in name.lower() for b in _NON_LM):
            names.append(name)
    # Fallback to bare suffixes if discovery somehow found nothing.
    return names or list(proj)


# A clean ChatML fallback template, only used if Kumru ships without a chat_template.
_CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def response_terminator_id(tokenizer):
    """The token that ends an assistant turn for THIS model family. Gemma ends
    turns with <end_of_turn> (not </s>); ChatML models with <|im_end|>; most
    others with eos. Test real VOCAB membership — convert_tokens_to_ids never
    returns None (it returns unk), and unk_token_id is None on Qwen, so the old
    guard was unreliable."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001
        vocab = {}
    for tok in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        if tok in vocab:
            return vocab[tok]
    return tokenizer.eos_token_id


def apply_ct(tokenizer, messages, **kwargs):
    """apply_chat_template that disables any forced <think> block (Gemma-4/Qwen3)
    and degrades gracefully if the template doesn't accept the kwarg."""
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def ct_ids(tokenizer, messages, **kwargs):
    """Return a flat list[int] of token ids from apply_chat_template, robust to
    transformers versions that return a BatchEncoding/dict or a batched list.
    Falls back to plain concatenation if the template itself errors."""
    try:
        out = apply_ct(tokenizer, messages, tokenize=True, **kwargs)
    except Exception:  # noqa: BLE001  — degrade instead of crashing the run
        text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        return list(tokenizer(text, add_special_tokens=True)["input_ids"])
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):   # unwrap batch dim
        out = out[0]
    return list(out)


def ct_tensor(tokenizer, messages, device=None, **kwargs):
    """Return (input_ids_tensor, attention_mask_or_None) for generation, robust to
    apply_chat_template returning a BatchEncoding/dict OR a bare tensor. Treating a
    BatchEncoding as a tensor (e.g. `.shape[1]`) raises a bare AttributeError."""
    enc = apply_ct(tokenizer, messages, return_tensors="pt", **kwargs)
    if hasattr(enc, "input_ids"):          # BatchEncoding
        input_ids, attn = enc.input_ids, getattr(enc, "attention_mask", None)
    elif isinstance(enc, dict):
        input_ids, attn = enc["input_ids"], enc.get("attention_mask")
    else:                                  # already a tensor
        input_ids, attn = enc, None
    if device is not None:
        input_ids = input_ids.to(device)
        if attn is not None:
            attn = attn.to(device)
    return input_ids, attn


def ensure_chat_and_pad(tokenizer):
    if tokenizer.chat_template is None:
        print("==> Tokenizer has no chat_template; applying a ChatML fallback.")
        tokenizer.chat_template = _CHATML
    if tokenizer.eos_token_id is None:
        sys.exit("ABORT: tokenizer has no eos_token; cannot teach the model to stop.")
    # Pad MUST be distinct from EOS, otherwise the stop signal is muddied and the
    # model's config.pad_token_id aliases EOS. Prefer unk only if it is itself
    # distinct from eos/bos; else add a dedicated pad token (+ resize later).
    if tokenizer.pad_token is None:
        unk_id = tokenizer.unk_token_id
        if (tokenizer.unk_token is not None
                and unk_id not in (tokenizer.eos_token_id, tokenizer.bos_token_id)):
            tokenizer.pad_token = tokenizer.unk_token
            print(f"==> pad_token set to unk_token ({tokenizer.unk_token!r}).")
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print("==> Added a dedicated <|pad|> token (embeddings will be resized).")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Last resort: force a dedicated pad so PAD never aliases EOS.
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print("==> pad_token aliased eos; replaced with dedicated <|pad|>.")
    tokenizer.padding_side = "right"
    return tokenizer


# ----------------------------------------------------------------------------
# Tokenisation with ASSISTANT-ONLY loss masking, BY CONSTRUCTION (not by diffing
# two renders). We render the prompt with the generation prompt, then append the
# assistant content tokens followed by a REAL EOS. The prompt span is masked to
# -100; only the assistant content + EOS carry loss. This guarantees:
#   * a clean mask boundary regardless of the chat template's quirks, and
#   * a real stop token in every supervised target (model learns to stop).
# ----------------------------------------------------------------------------
def make_encoder(tokenizer, max_len):
    eot_id = response_terminator_id(tokenizer)   # family-aware turn terminator
    errs = {"n": 0}

    def encode(example):
        try:
            msgs = example["messages"]
            prompt_ids = ct_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
            response_ids = list(tokenizer(
                msgs[-1]["content"], add_special_tokens=False)["input_ids"]) + [eot_id]
        except Exception as e:  # noqa: BLE001 — drop the row, never kill the run
            errs["n"] += 1
            if errs["n"] <= 3:
                print(f"==> encode skip ({type(e).__name__}: {e}) — row dropped.")
            return {"input_ids": [eot_id], "attention_mask": [1],
                    "labels": [-100], "n_supervised": 0}

        input_ids = list(prompt_ids) + list(response_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)

        # Left-truncate to keep the (tail) assistant span if over length.
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "n_supervised": sum(1 for t in labels if t != -100),
        }
    return encode


def build_datasets(rows, tokenizer):
    from datasets import Dataset
    encode = make_encoder(tokenizer, CONFIG["max_seq_len"])
    ds = Dataset.from_list(rows).map(
        encode, remove_columns=["messages"], desc="tokenizing")
    # Telemetry: cheap guard against silent mask corruption (C1/C2).
    sup = sorted(ds["n_supervised"])
    median = sup[len(sup) // 2]
    print(f"==> supervised tokens/example: min={sup[0]} median={median} max={sup[-1]}")
    if sup[0] <= 1:
        print("==> WARNING: some example(s) have <=1 supervised token — check the "
              "chat template / data; the assistant target may be getting truncated.")
    ds = ds.filter(lambda ex: ex["n_supervised"] > 0).remove_columns(["n_supervised"])
    n_eval = max(1, int(len(ds) * CONFIG["eval_fraction"])) if len(ds) > 10 else 0
    if n_eval:
        ds = ds.train_test_split(test_size=n_eval, seed=CONFIG["seed"])
        return ds["train"], ds["test"]
    return ds, None


# ----------------------------------------------------------------------------
# Training (smoke -> full)
# ----------------------------------------------------------------------------
def build_trainer(model, tokenizer, train_ds, eval_ds, backend, max_steps=-1):
    import torch
    from transformers import (DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding="longest", label_pad_token_id=-100, return_tensors="pt")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not bf16_ok:
        print("==> WARNING: bf16 unavailable; falling back to fp16 (QLoRA is more "
              "prone to loss spikes/NaN under fp16 — watch the smoke loss).")
    full_run = max_steps < 0
    do_eval = full_run and eval_ds is not None
    # Rigor: on the full run, evaluate periodically and KEEP THE BEST checkpoint by
    # eval_loss (guards against the last-epoch overfit the light run couldn't see).
    common = dict(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        num_train_epochs=CONFIG["epochs"] if full_run else 1,
        max_steps=max_steps,
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=bf16_ok, fp16=not bf16_ok,
        gradient_checkpointing=False,   # already enabled at model load
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=CONFIG["seed"],
    )
    if do_eval:
        # Cadence scales with the run: ~10 evals total (not a fixed 25 that would
        # over-eval at hardcore scale and never fire on tiny data).
        steps_per_epoch = max(1, len(train_ds) // (CONFIG["batch_size"] * CONFIG["grad_accum"]))
        total_steps = steps_per_epoch * max(1, int(CONFIG["epochs"]))
        eval_every = CONFIG.get("eval_steps") or max(10, total_steps // 10)
        common.update(
            eval_strategy="steps", save_strategy="steps",
            eval_steps=eval_every, save_steps=eval_every,
            save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False)
        print(f"==> eval/checkpoint every {eval_every} steps (~{total_steps} total).")
    else:
        common.update(eval_strategy="no",
                      save_strategy=("epoch" if full_run else "no"))
    args = TrainingArguments(**common)
    return Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=(eval_ds if do_eval else None), data_collator=collator,
    )


def _losses_finite(trainer):
    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    bad = [x for x in losses if not math.isfinite(x)]
    return losses, bad


def train(model, tokenizer, train_ds, eval_ds, backend, smoke_only, synthetic_run=False):
    # --- Smoke: ~20 steps. Pure plumbing test on the real model: confirm it runs
    #     and the loss is finite. (These few updates barely move the weights; the
    #     full run below carries the real training — smoke is not a warmup.) ---
    print(f"\n==> SMOKE RUN ({CONFIG['smoke_steps']} steps; plumbing only) ...")
    smoke = build_trainer(model, tokenizer, train_ds, eval_ds, backend,
                          max_steps=CONFIG["smoke_steps"])
    res = smoke.train()
    losses, bad = _losses_finite(smoke)
    if bad:
        sys.exit("ABORT: smoke loss is NaN/inf — check data/precision before full run.")
    if losses:
        print(f"    smoke loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"    smoke ok (runtime {res.metrics.get('train_runtime', 0):.1f}s).")

    if smoke_only:
        print("==> --smoke-only set; stopping before the full run.")
        return

    # --- Full run ---
    print("\n==> FULL RUN ...")
    full = build_trainer(model, tokenizer, train_ds, eval_ds, backend, max_steps=-1)
    full.train()
    _, bad = _losses_finite(full)
    if bad:
        print("==> WARNING: non-finite loss observed during the full run.")
    # Assert best-checkpoint selection actually happened (guards a silent quantized
    # reload no-op that would ship the LAST-step adapter instead of the best).
    if eval_ds is not None:
        bm = getattr(full.state, "best_model_checkpoint", None)
        if bm is None:
            print("==> WARNING: load_best_model_at_end did NOT restore a best adapter "
                  "(best_model_checkpoint is None) — you have the LAST-step adapter.")
        else:
            print(f"==> Best adapter by eval_loss: {bm} (best={full.state.best_metric})")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    # Remove intermediate checkpoint-*/ dirs so the root holds EXACTLY the best
    # adapter — otherwise eval could resolve a non-best checkpoint's adapter_config.
    import glob
    import shutil
    for ckpt in glob.glob(os.path.join(CONFIG["output_dir"], "checkpoint-*")):
        if os.path.isdir(ckpt):
            shutil.rmtree(ckpt, ignore_errors=True)
    # Provenance stamp — read by evaluate.py to choose the clinical vs research gate.
    with open(os.path.join(CONFIG["output_dir"], "PROVENANCE.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "base_model": CONFIG["base_model"],
            "synthetic": synthetic_run,
            "note": ("Trained on machine-generated (LLM-distilled) data grounded in "
                     "open literature; NOT clinician-reviewed; research prototype, "
                     "NOT for clinical use." if synthetic_run else
                     "Trained on clinician-reviewed data."),
        }, fh, ensure_ascii=False, indent=2)
    print(f"==> Saved LoRA adapter to {CONFIG['output_dir']}"
          + ("  [SYNTHETIC — research prototype]" if synthetic_run else ""))


# ----------------------------------------------------------------------------
# Sanity check: tuned (and, on the HF path, base via disable_adapter) side by
# side. Synthetic, NON-CLINICAL plumbing vignettes with PLACEHOLDER passages —
# the *safe* behaviour on an empty passage is an empty kaynak + everything under
# eksik_veriler. We assert valid JSON and FLAG fabricated grounding.
# ----------------------------------------------------------------------------
SANITY_SYSTEM = GUARDRAIL_SYSTEM   # same canonical guardrail used at train time
SANITY_VIGNETTES = [
    "Hasta bağlamı: [sentetik test vakası A]. Kılavuz pasajı: [boş — pasaj verilmedi].",
    "Hasta bağlamı: [sentetik test vakası B]. Kılavuz pasajı: [boş — pasaj verilmedi].",
    "Hasta bağlamı: [sentetik test vakası C]. Kılavuz pasajı: [boş — pasaj verilmedi].",
    "Hasta bağlamı: [sentetik test vakası D]. Kılavuz pasajı: [boş — pasaj verilmedi].",
    "Hasta bağlamı: [sentetik test vakası E]. Kılavuz pasajı: [boş — pasaj verilmedi].",
]


def _generate(model, tokenizer, user_text):
    import torch
    msgs = [{"role": "system", "content": SANITY_SYSTEM},
            {"role": "user", "content": user_text}]
    dev = model.get_input_embeddings().weight.device
    input_ids, attn = ct_tensor(tokenizer, msgs, device=dev, add_generation_prompt=True)
    kw = {} if attn is None else {"attention_mask": attn}
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=256, do_sample=False,
                             eos_token_id=response_terminator_id(tokenizer),
                             pad_token_id=tokenizer.pad_token_id, **kw)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


def _flag_ungrounded(tuned_text):
    """On a placeholder (empty) passage, a fabricated kaynak or suggestion is a
    grounding failure. Returns a warning string, or '' if it looks safe."""
    try:
        card = json.loads(tuned_text)
    except Exception:  # noqa: BLE001
        return "output is not valid card JSON"
    if str(card.get("kaynak", "")).strip():
        return "FABRICATED kaynak from an empty passage (grounding failure)"
    if card.get("onerilen_sorular") or card.get("onerilen_tetkikler"):
        return "suggestions produced from an empty passage (ungrounded)"
    return ""


def sanity_check(model, tokenizer, backend):
    print("\n==> SANITY CHECK (synthetic placeholder vignettes) ...")
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception:  # noqa: BLE001
            model.eval()
    else:
        model.eval()   # HF path: don't import Unsloth (avoids late-patch warnings)

    for i, v in enumerate(SANITY_VIGNETTES, 1):
        print(f"\n----- vignette {i} -----")
        # disable_adapter() is reliable for HF/PEFT; on Unsloth it does not
        # faithfully reproduce the base forward pass, so we skip it honestly.
        if backend == "hf":
            try:
                with model.disable_adapter():
                    base = _generate(model, tokenizer, v)
            except Exception as e:  # noqa: BLE001
                base = f"[base generation skipped: {e}]"
            print(f"[BASE ]\n{base}\n")
        else:
            print("[BASE ] (skipped on Unsloth path — disable_adapter() is unreliable here)\n")
        tuned = _generate(model, tokenizer, v)
        warn = _flag_ungrounded(tuned)
        print(f"[TUNED]\n{tuned}")
        if warn:
            print(f"  ⚠ GROUNDING WARNING: {warn} — escalate to cdss-safety-redteam.")
    print("\n==> Sanity check done. NOTE: passing this is NOT clinical safety. A run "
          "is 'ready to EVALUATE', not 'ready to use'. Required before use: held-out "
          "clinician eval + missing-data/citation red-team must pass critical cases.")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Kumru-2B for TR neoperi cards.")
    ap.add_argument("data", nargs="?", default=None, help="path to reviewed training .jsonl")
    ap.add_argument("run_name", nargs="?", default=None, help="optional run name (adapter subdir)")
    ap.add_argument("--install-deps", action="store_true", help="pip-install deps and exit")
    ap.add_argument("--smoke-only", action="store_true", help="run ~20 steps then stop")
    ap.add_argument("--base-model", default=None, help="override base model id")
    ap.add_argument("--output-dir", default=None, help="override adapter output dir")
    ap.add_argument("--no-unsloth", action="store_true", help="force HF path (default)")
    ap.add_argument("--unsloth", action="store_true",
                    help="opt into Unsloth (needs Triton+gcc and a compatible torch)")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="also accept machine-generated rows (provenance.source=='auto'); "
                         "the run is labelled synthetic and cannot earn a clinical RELEASE_OK")
    # --- Scale / capacity knobs (override the inline CONFIG) ---
    ap.add_argument("--epochs", type=float, default=None, help="training epochs")
    ap.add_argument("--lora-r", type=int, default=None, help="LoRA rank (capacity)")
    ap.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    ap.add_argument("--lr", type=float, default=None, help="learning rate")
    ap.add_argument("--max-seq-len", type=int, default=None, help="max sequence length")
    ap.add_argument("--batch-size", type=int, default=None, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps")
    ap.add_argument("--attn-impl", default=None,
                    help="attention impl for the HF path, e.g. 'eager' (recommended for Gemma)")
    args = ap.parse_args()

    if args.install_deps:
        install_deps()
        return

    if args.base_model:
        CONFIG["base_model"] = args.base_model
    if args.no_unsloth:
        CONFIG["use_unsloth"] = False
    if getattr(args, "unsloth", False):
        CONFIG["use_unsloth"] = True
    # Apply capacity/scale overrides.
    for cli, key in (("epochs", "epochs"), ("lora_r", "lora_r"), ("lora_alpha", "lora_alpha"),
                     ("lr", "learning_rate"), ("max_seq_len", "max_seq_len"),
                     ("batch_size", "batch_size"), ("grad_accum", "grad_accum"),
                     ("attn_impl", "attn_impl")):
        val = getattr(args, cli)
        if val is not None:
            CONFIG[key] = val
    # Non-Kumru families are safest on the HF path (Unsloth lags new archs).
    if "kumru" not in CONFIG["base_model"].lower():
        CONFIG["use_unsloth"] = False
    if args.output_dir:
        CONFIG["output_dir"] = args.output_dir
    elif args.run_name:
        CONFIG["output_dir"] = os.path.join("models", f"kumru-neoperi-lora-{args.run_name}")

    data_path = args.data or CONFIG["default_data"]

    print("=" * 78)
    print(f">>> neoperi code version: {NEOPERI_VERSION} <<<")
    print("Kumru-2B  TR neonatology/perinatology  QLoRA fine-tune")
    print(f"  base={CONFIG['base_model']}  out={CONFIG['output_dir']}  data={data_path}")
    print(f"  epochs={CONFIG['epochs']}  lora_r={CONFIG['lora_r']}  "
          f"lora_alpha={CONFIG['lora_alpha']}  lr={CONFIG['learning_rate']}  "
          f"max_seq_len={CONFIG['max_seq_len']}  bs={CONFIG['batch_size']}x{CONFIG['grad_accum']}")
    print("=" * 78)

    check_gpu()
    rows, synthetic_run = load_data(data_path, allow_synthetic=args.allow_synthetic)
    model, tokenizer, backend = load_model_and_tokenizer()
    tokenizer = ensure_chat_and_pad(tokenizer)
    # Apply to BOTH backends: if a pad token was added, the embedding table must
    # be resized or pad ids index out of bounds at runtime (latent crash).
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            print(f"==> Resizing embeddings to {len(tokenizer)} (new pad token).")
            model.resize_token_embeddings(len(tokenizer))
    train_ds, eval_ds = build_datasets(rows, tokenizer)
    print(f"==> train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}  backend={backend}")

    train(model, tokenizer, train_ds, eval_ds, backend, smoke_only=args.smoke_only,
          synthetic_run=synthetic_run)
    if not args.smoke_only:
        sanity_check(model, tokenizer, backend)


if __name__ == "__main__":
    main()
