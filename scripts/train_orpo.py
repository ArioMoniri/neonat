#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_orpo.py — Phase-2 PREFERENCE alignment (ORPO) after the QLoRA SFT
========================================================================
SFT (train_lora.py) teaches the TURKISH neonatology/perinatology suggestion-card
FORMAT. This second phase uses ORPO (odds-ratio preference optimisation) to push
the model AWAY from the two lethal card failures and TOWARD the safe card:

    chosen   = a validate_card-valid GROUNDED or REFUSAL card (the safe answer)
    rejected = a HALLUCINATED-CITATION card (grounded shape, invented `kaynak`
               on an ungroundable prompt) OR a SHOULD-HAVE-REFUSED / prescribing
               card (an imperative drug/dose/order injected into the free text)

ORPO is reference-free (no frozen ref model / no separate reward model): one 4-bit
NF4 base + one LoRA adapter, a monolithic loss = SFT NLL + beta * odds-ratio term.

Mirrors train_lora.py exactly where it matters: same QLoRA/QDoRA NF4 loader, same
hf_dtype_kwargs dtype shim, the same fail-closed data gate, the SAME validate_card
(imported via importlib as TL) so "valid card" means one thing across SFT/ORPO/eval,
and the same --allow-synthetic provenance gate.

--------------------------------------------------------------------------------
QUICK START (on the GPU box, inside the venv):
    # A) derive preferences from the SFT set and align, continuing the SFT adapter:
    python scripts/train_orpo.py --from-sft data/processed/task_sft.synth.jsonl \
        --build-prefs --allow-synthetic \
        --base-model vngrs-ai/Kumru-2B \
        --adapter models/kumru-neoperi-synth \
        --output-dir models/kumru-neoperi-orpo

    # B) train from a ready preference file {prompt, chosen, rejected} per line:
    python scripts/train_orpo.py --data data/processed/task_prefs.jsonl \
        --base-model vngrs-ai/Kumru-2B --output-dir models/kumru-neoperi-orpo

--------------------------------------------------------------------------------
SAFETY NOTE:
  A converged ORPO loss is NOT clinical safety. This is a RESEARCH prototype on
  SYNTHETIC data; the adapter is "ready to EVALUATE", never "ready to use". The
  clinician held-out eval + missing-data/citation red-team (evaluate.py) remain
  the real gate. Preference data derived from synthetic cards stays synthetic.
================================================================================
"""

import argparse
import copy
import gc
import importlib.util
import json
import os
import sys

NEOPERI_ORPO_VERSION = "2026-07-16-orpo-preference-phase2"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_train_module():
    """Import train_lora.py as TL to reuse validate_card, hf_dtype_kwargs, the
    chat-template (apply_ct) helpers, the LoRA-target discovery, and the fail-closed
    row validator — one source of truth shared with SFT and eval."""
    path = os.path.join(_HERE, "train_lora.py")
    spec = importlib.util.spec_from_file_location("train_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TL = _load_train_module()

# ----------------------------------------------------------------------------
# INLINE CONFIG (CLI overrides these; defaults follow the spec)
# ----------------------------------------------------------------------------
CONFIG = {
    "base_model":     "vngrs-ai/Kumru-2B",
    "adapter":        None,            # SFT adapter to CONTINUE from (optional)
    "output_dir":     "models/kumru-neoperi-orpo",
    "max_seq_len":    2048,
    "lora_r":         16,
    "lora_alpha":     16,
    "lora_dropout":   0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "epochs":         1,
    "learning_rate":  6.0e-6,
    "beta":           0.1,             # ORPO odds-ratio weight (lambda)
    "warmup_ratio":   0.1,
    "weight_decay":   0.0,
    "batch_size":     2,
    "grad_accum":     8,
    "use_dora":       True,            # QDoRA on the 4-bit base (matches SFT)
    "eval_fraction":  0.1,
    "seed":           42,
}


# ----------------------------------------------------------------------------
# Preference construction
# ----------------------------------------------------------------------------
# A fabricated citation string for the hallucinated-grounding rejected card. It is
# deliberately obviously-invented; the point is that NO real passage backs it.
_FAKE_KAYNAK = "Uydurma Kılavuz Derlemesi 2021, Bölüm 3, s. 42"


def _corrupt_to_rejected(card):
    """Programmatically turn a SAFE card (chosen) into a REJECTED negative.

    refusal -> a hallucinated-citation GROUNDED card: invent a kaynak + suggestions
               on what was an ungroundable prompt (the false-grounding failure).
    grounded -> inject an imperative prescription/order into the free text so it
               trips violates_action_policy (the should-have-suggested-not-ordered
               failure). Returns a JSON string DIFFERENT from the chosen card."""
    c = copy.deepcopy(card)
    karar = c.get("karar", "grounded")
    if karar == "refusal":
        c["karar"] = "grounded"
        c["kaynak"] = _FAKE_KAYNAK                       # fabricated grounding
        c["onerilen_tetkikler"] = ["tam kan sayımı", "CRP", "kan kültürü"]
        c["onerilen_sorular"] = c.get("onerilen_sorular") or ["Ateş öyküsü var mı?"]
        c["eksik_veriler"] = []
        c["uyari"] = ("Bulgular sepsis ile uyumludur; ampirik antibiyotik "
                      "başlanması uygundur.")
        c.pop("gerekce", None)
    else:
        # keep grounded shape but smuggle in an order/dose (unsafe on any case)
        c["uyari"] = "Ampisilin 50 mg/kg IV başlayın ve fototerapiye geçin."
        c["onerilen_eylem"] = {"verb": "hekime_danis",
                               "aciklama": "Gentamisin 5 mg/kg hemen başlayın."}
    out = json.dumps(c, ensure_ascii=False)
    if out == json.dumps(card, ensure_ascii=False):     # never emit chosen==rejected
        c["uyari"] = (str(c.get("uyari", "")) +
                      " Kesin tanı sepsis; 40 mg/kg dozunda tedaviye başlayın.")
        out = json.dumps(c, ensure_ascii=False)
    return out


def build_prefs_from_sft(path, allow_synthetic=False):
    """Derive a preference set from an SFT chat JSONL. Each valid row becomes one
    (prompt, chosen, rejected) triple: chosen = the row's validate_card-valid
    assistant card, rejected = _corrupt_to_rejected(chosen). Fail-closed — a row
    that would not pass the SFT gate (unreviewed / malformed / ungrounded) is a HARD
    ABORT, exactly like train_lora.load_data (with --allow-synthetic mirroring)."""
    if not os.path.exists(path):
        sys.exit(f"ABORT: --from-sft file not found: {path}")
    rows, rejects, n_synth = [], [], 0
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
            ok, reason, clean = TL._validate_row(obj, allow_synthetic=allow_synthetic)
            if not ok:
                rejects.append((i, reason))
                continue
            if TL._is_synthetic(obj):
                n_synth += 1
            msgs = clean["messages"]
            chosen = str(msgs[-1]["content"])
            try:
                card = json.loads(chosen)
            except json.JSONDecodeError:
                rejects.append((i, "chosen assistant content is not card JSON"))
                continue
            rejected = _corrupt_to_rejected(card)
            rows.append({"prompt": msgs[:-1], "chosen": chosen, "rejected": rejected})
    if rejects:
        preview = "\n".join(f"       line {ln}: {why}" for ln, why in rejects[:15])
        more = "" if len(rejects) <= 15 else f"\n       ... and {len(rejects) - 15} more"
        sys.exit(f"ABORT: {len(rejects)} SFT row(s) failed the fail-closed gate while "
                 f"building preferences (unreviewed / malformed / ungrounded). Only "
                 f"clinician-approved (or, with --allow-synthetic, machine-generated "
                 f"provenance), schema-valid, grounded rows may seed a chosen card.\n"
                 f"{preview}{more}")
    if not rows:
        sys.exit("ABORT: no valid rows to build preferences from.")
    synthetic_run = n_synth > 0
    print(f"==> Built {len(rows)} preference pair(s) from SFT "
          f"[{n_synth} synthetic-seeded, {len(rows) - n_synth} reviewed-seeded].")
    return rows, synthetic_run


def load_pref_data(path):
    """Load a ready preference JSONL: one {prompt, chosen, rejected} object per line.
    `prompt` may be a rendered string OR a list of chat messages. Fail-closed: chosen
    must parse+validate as a suggestion-card, chosen!=rejected, all fields present."""
    if not os.path.exists(path):
        sys.exit(f"ABORT: --data preference file not found: {path}")
    rows, rejects = [], []
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
            prompt, chosen, rejected = obj.get("prompt"), obj.get("chosen"), obj.get("rejected")
            if prompt is None or not isinstance(chosen, str) or not isinstance(rejected, str):
                rejects.append((i, "need prompt + string chosen + string rejected"))
                continue
            if chosen.strip() == rejected.strip():
                rejects.append((i, "chosen == rejected (no preference signal)"))
                continue
            try:
                ok, reason = TL.validate_card(json.loads(chosen))
            except json.JSONDecodeError:
                ok, reason = False, "chosen is not card JSON"
            if not ok:
                rejects.append((i, f"chosen is not a valid card: {reason}"))
                continue
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if rejects:
        preview = "\n".join(f"       line {ln}: {why}" for ln, why in rejects[:15])
        more = "" if len(rejects) <= 15 else f"\n       ... and {len(rejects) - 15} more"
        sys.exit(f"ABORT: {len(rejects)} preference row(s) failed the fail-closed gate.\n"
                 f"{preview}{more}")
    if not rows:
        sys.exit("ABORT: no valid preference rows found.")
    print(f"==> Loaded {len(rows)} preference pair(s) from {path}.")
    # External pref files carry no provenance guarantee -> treat as synthetic/research.
    return rows, True


# ----------------------------------------------------------------------------
# Render prompts + split into an ORPO Dataset (standard string format)
# ----------------------------------------------------------------------------
def _render_prompt(tokenizer, prompt):
    """A chat-message list -> a prompt STRING with the generation prompt appended;
    a string passes through unchanged. Uses TL.apply_ct (disables forced <think>)."""
    if isinstance(prompt, str):
        return prompt
    text = TL.apply_ct(tokenizer, prompt, tokenize=False, add_generation_prompt=True)
    return text if isinstance(text, str) else str(text)


def build_datasets(rows, tokenizer):
    from datasets import Dataset
    recs = [{"prompt": _render_prompt(tokenizer, r["prompt"]),
             "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    ds = Dataset.from_list(recs)
    n_eval = max(1, int(len(ds) * CONFIG["eval_fraction"])) if len(ds) > 10 else 0
    if n_eval:
        split = ds.train_test_split(test_size=n_eval, seed=CONFIG["seed"])
        return split["train"], split["test"]
    return ds, None


# ----------------------------------------------------------------------------
# Model + tokenizer: QLoRA NF4 loader mirroring train_lora.py (HF + bitsandbytes)
# ----------------------------------------------------------------------------
def load_base_and_tokenizer():
    """4-bit NF4 base + tokenizer. If CONFIG['adapter'] is set, attach that SFT
    adapter as a TRAINABLE PeftModel and CONTINUE it (peft_config=None downstream);
    otherwise return the bare quantized base and let ORPOTrainer apply a fresh LoRA."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import prepare_model_for_kbit_training

    cfg = CONFIG
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok_src = cfg["adapter"] or cfg["base_model"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    except Exception as e:  # noqa: BLE001
        print(f"==> AutoTokenizer failed ({e}); trying AutoProcessor.tokenizer.")
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(tok_src).tokenizer

    print("==> Loading 4-bit NF4 base via Hugging Face + bitsandbytes.")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], quantization_config=bnb, device_map="auto",
            **TL.hf_dtype_kwargs())
    except Exception as e:  # noqa: BLE001
        print(f"==> AutoModelForCausalLM failed ({type(e).__name__}); trying "
              "AutoModelForImageTextToText (multimodal, e.g. Gemma 4).")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["base_model"], quantization_config=bnb, device_map="auto",
            **TL.hf_dtype_kwargs())
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if cfg["adapter"]:
        from peft import PeftModel
        if not os.path.exists(cfg["adapter"]):
            sys.exit(f"ABORT: --adapter dir not found: {cfg['adapter']}")
        print(f"==> Continuing from SFT adapter (trainable): {cfg['adapter']}")
        model = PeftModel.from_pretrained(model, cfg["adapter"], is_trainable=True)
    return model, tokenizer


def build_lora_config():
    """Fresh LoRA/QDoRA config handed to ORPOTrainer when NOT continuing an adapter.
    Mirrors train_lora.py: rsLoRA at r>=32, QDoRA by default, drop-retry so older
    peft (no use_dora/use_rslora) still constructs a config."""
    from peft import LoraConfig
    cfg = CONFIG
    lora_kw = dict(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=list(cfg["target_modules"]), bias="none", task_type="CAUSAL_LM",
    )
    if cfg["lora_r"] >= 32:
        lora_kw["use_rslora"] = True
    if cfg.get("use_dora", True):
        lora_kw["use_dora"] = True
    lora, last_err = None, None
    for drop in ([], ["use_dora"], ["use_dora", "use_rslora"]):
        kw = {k: v for k, v in lora_kw.items() if k not in drop}
        try:
            lora = LoraConfig(**kw)
            if drop:
                print(f"==> peft rejected {drop}; retried without.")
            break
        except TypeError as e:
            last_err = e
    if lora is None:
        raise last_err
    return lora


# ----------------------------------------------------------------------------
# ORPO trainer construction (trl) + version self-heal
# ----------------------------------------------------------------------------
def _require_trl():
    try:
        from trl import ORPOConfig, ORPOTrainer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit("ABORT: trl (with ORPOTrainer) is required for Phase-2 ORPO but is "
                 f"unavailable ({type(e).__name__}: {e}). Install it, e.g.:\n"
                 "       pip install -U 'trl>=0.12'")


def _make_orpo_trainer(model, tokenizer, args, train_ds, eval_ds, peft_config):
    from trl import ORPOTrainer
    base_kw = dict(model=model, args=args, train_dataset=train_ds,
                   eval_dataset=eval_ds, peft_config=peft_config)
    # trl renamed `tokenizer` -> `processing_class` around 0.12; try new then old.
    last = None
    for key in ("processing_class", "tokenizer"):
        try:
            return ORPOTrainer(**base_kw, **{key: tokenizer})
        except TypeError as e:
            last = e
    raise last


def build_orpo_trainer(model, tokenizer, train_ds, eval_ds, peft_config):
    import torch
    from trl import ORPOConfig

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not bf16_ok:
        print("==> WARNING: bf16 unavailable; falling back to fp16 (QLoRA is more "
              "prone to loss spikes under fp16 — watch the ORPO loss).")
    do_eval = eval_ds is not None
    common = dict(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        beta=CONFIG["beta"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        max_length=CONFIG["max_seq_len"],
        max_prompt_length=max(256, CONFIG["max_seq_len"] // 2),
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=bf16_ok, fp16=not bf16_ok,
        gradient_checkpointing=False,   # already enabled at model load (prepare_model_for_kbit_training);
                                        # mirrors train_lora.py — one source of truth, no double-enable
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=CONFIG["seed"],
        remove_unused_columns=False,
    )
    if do_eval:
        common.update(eval_strategy="epoch", save_strategy="epoch",
                      save_total_limit=2)
    else:
        common.update(eval_strategy="no", save_strategy="epoch")
    args = ORPOConfig(**common)

    try:
        return _make_orpo_trainer(model, tokenizer, args, train_ds, eval_ds, peft_config)
    except (ValueError, RuntimeError, NotImplementedError) as e:
        # Some peft/base combos reject DoRA on a quantized base at apply time.
        if peft_config is not None and getattr(peft_config, "use_dora", False):
            print(f"==> DoRA apply failed ({type(e).__name__}: {e}); retrying plain LoRA.")
            peft_config.use_dora = False
            return _make_orpo_trainer(model, tokenizer, args, train_ds, eval_ds, peft_config)
        raise


# ----------------------------------------------------------------------------
# Provenance stamp (read by evaluate.py to pick the clinical vs research gate)
# ----------------------------------------------------------------------------
def write_provenance(synthetic_run):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["output_dir"], "PROVENANCE.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "base_model": CONFIG["base_model"],
            "phase": "orpo-preference",
            "sft_adapter": CONFIG["adapter"],
            "beta": CONFIG["beta"],
            "synthetic": bool(synthetic_run),
            "note": ("ORPO preference alignment over machine-generated / "
                     "programmatically-corrupted preference pairs; research "
                     "prototype, NOT clinician-reviewed, NOT for clinical use."
                     if synthetic_run else
                     "ORPO preference alignment over clinician-reviewed cards."),
        }, fh, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Phase-2 ORPO preference alignment for TR neoperi cards.")
    ap.add_argument("--data", default=None,
                    help="preference JSONL {prompt, chosen, rejected} per line")
    ap.add_argument("--from-sft", default=None,
                    help="SFT chat JSONL to derive preferences from (with --build-prefs)")
    ap.add_argument("--build-prefs", action="store_true",
                    help="derive the preference set from --from-sft by corrupting each "
                         "row's chosen card into a rejected negative")
    ap.add_argument("--base-model", default=None, help="override base model id")
    ap.add_argument("--adapter", default=None,
                    help="SFT LoRA adapter to CONTINUE from (optional)")
    ap.add_argument("--output-dir", default=None, help="override adapter output dir")
    ap.add_argument("--epochs", type=float, default=None, help="training epochs (default 1)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default 6e-6)")
    ap.add_argument("--beta", type=float, default=None, help="ORPO odds-ratio weight (default 0.1)")
    ap.add_argument("--max-seq-len", type=int, default=None, help="max sequence length (default 2048)")
    ap.add_argument("--batch-size", type=int, default=None, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps")
    ap.add_argument("--lora-r", type=int, default=None, help="LoRA rank (fresh adapter path)")
    ap.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (fresh adapter path)")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="also accept machine-generated rows (provenance.source=='auto'); "
                         "the run is labelled synthetic and cannot earn a clinical RELEASE_OK")
    args = ap.parse_args()

    # Apply overrides.
    if args.base_model:
        CONFIG["base_model"] = args.base_model
    if args.adapter:
        CONFIG["adapter"] = args.adapter
    if args.output_dir:
        CONFIG["output_dir"] = args.output_dir
    for cli, key in (("epochs", "epochs"), ("lr", "learning_rate"), ("beta", "beta"),
                     ("max_seq_len", "max_seq_len"), ("batch_size", "batch_size"),
                     ("grad_accum", "grad_accum"), ("lora_r", "lora_r"),
                     ("lora_alpha", "lora_alpha")):
        val = getattr(args, cli)
        if val is not None:
            CONFIG[key] = val

    # Resolve the preference source (exactly one).
    if args.build_prefs or args.from_sft:
        if not args.from_sft:
            sys.exit("ABORT: --build-prefs requires --from-sft PATH.")
        if args.data:
            sys.exit("ABORT: pass EITHER --data or --from-sft/--build-prefs, not both.")
        pref_source = ("build", args.from_sft)
    elif args.data:
        pref_source = ("data", args.data)
    else:
        sys.exit("ABORT: provide --data <pref.jsonl>  OR  --from-sft <sft.jsonl> --build-prefs.")

    print("=" * 78)
    print(f">>> neoperi ORPO code version: {NEOPERI_ORPO_VERSION} <<<")
    print("Phase-2 ORPO preference alignment  (TR neonatology/perinatology cards)")
    print(f"  base={CONFIG['base_model']}  adapter={CONFIG['adapter']}  "
          f"out={CONFIG['output_dir']}")
    print(f"  epochs={CONFIG['epochs']}  lr={CONFIG['learning_rate']}  "
          f"beta={CONFIG['beta']}  max_seq_len={CONFIG['max_seq_len']}  "
          f"bs={CONFIG['batch_size']}x{CONFIG['grad_accum']}")
    print("=" * 78)

    _require_trl()

    # Data (fail-closed).
    if pref_source[0] == "build":
        rows, synthetic_run = build_prefs_from_sft(
            pref_source[1], allow_synthetic=args.allow_synthetic)
    else:
        rows, synthetic_run = load_pref_data(pref_source[1])
    if synthetic_run:
        print("==> SYNTHETIC RUN: preference data is machine-generated / programmatically "
              "corrupted and NOT clinician-reviewed. The resulting adapter is a research "
              "prototype and cannot earn a clinical RELEASE_OK.")

    # Model + tokenizer.
    model, tokenizer = load_base_and_tokenizer()
    tokenizer = TL.ensure_chat_and_pad(tokenizer)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            print(f"==> Resizing embeddings to {len(tokenizer)} (new pad token).")
            model.resize_token_embeddings(len(tokenizer))

    train_ds, eval_ds = build_datasets(rows, tokenizer)
    print(f"==> train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}")

    # A fresh LoRA is applied by ORPOTrainer ONLY when we are not continuing an adapter.
    peft_config = None if CONFIG["adapter"] else build_lora_config()

    print("\n==> ORPO RUN ...")
    trainer = build_orpo_trainer(model, tokenizer, train_ds, eval_ds, peft_config)
    trainer.train()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    write_provenance(synthetic_run)
    print(f"==> Saved ORPO adapter to {CONFIG['output_dir']}"
          + ("  [SYNTHETIC — research prototype]" if synthetic_run else ""))
    print("==> NOTE: a converged ORPO loss is NOT clinical safety. Run evaluate.py "
          "(held-out clinician eval + missing-data/citation red-team) before any use.")

    # Explicit teardown so a chained multi-model run reclaims H200 VRAM cleanly.
    del trainer
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
