#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_encoder.py — a Turkish neonatology/perinatology ENCODER for the retrieval/NER
side of the CDSS (NOT the card generator).
================================================================================
Two paths:
  • DOMAIN-ADAPTIVE (default, recommended): continue MLM pretraining of an existing
    Turkish encoder (BERTurk / TabiBERT-ModernBERT) on the neoperi corpus. The
    literature (BioBERTurk, TurkRadBERT) shows this BEATS training from scratch in a
    low-resource domain — you keep the base's Turkish competence and specialize it.
  • FROM-SCRATCH (--from-scratch): initialize a *modern-architecture* encoder
    (RoPE/GeGLU via ModernBERT config) with random weights and MLM-train it. Honest
    note: with a small corpus this underperforms domain-adaptive; use only to
    experiment with architecture. (You CANNOT graft a new attention onto a
    pretrained decoder like Kumru — architecture is fixed at pretraining time.)

Input text = data/corpus/passages.jsonl (build a big one first with the hfds +
literature sources) and/or extra HF datasets.

Usage (GPU box, inside venv):
  python train_encoder.py --corpus data/corpus/passages.jsonl \
      --base dbmdz/bert-base-turkish-cased --epochs 3 --out models/neoperi-encoder
  python train_encoder.py --corpus ... --from-scratch --base answerdotai/ModernBERT-base
  python train_encoder.py --selftest    # offline: 1-step CPU smoke on tiny text
"""
import argparse
import json
import os
import sys


def load_texts(corpus, extra_hf):
    texts = []
    if corpus and os.path.exists(corpus):
        for line in open(corpus, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = p.get("passage") or p.get("text") or ""
            if len(t) > 40:
                texts.append(t)
    for ds_id in (extra_hf or []):
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_id, split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i >= 20000:
                    break
                t = next((v for v in ex.values() if isinstance(v, str) and len(v) > 40), None)
                if t:
                    texts.append(t)
        except Exception as e:  # noqa: BLE001
            print(f"==> extra HF dataset {ds_id} skipped: {e}")
    return texts


def main():
    ap = argparse.ArgumentParser(description="Domain-adaptive Turkish neoperi encoder (MLM).")
    ap.add_argument("--corpus", default="data/corpus/passages.jsonl")
    ap.add_argument("--extra-hf", action="append", default=[], help="extra HF dataset id(s)")
    ap.add_argument("--base", default="dbmdz/bert-base-turkish-cased",
                    help="encoder base (BERTurk default; use answerdotai/ModernBERT-base "
                         "for a MODERN arch = RoPE + GeGLU + local/global attention)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random-init the encoder from --base's config (keep tokenizer). "
                         "Pair with a ModernBERT base for a modern-architecture from-zero encoder.")
    ap.add_argument("--hidden", type=int, default=0, help="from-scratch: override hidden size")
    ap.add_argument("--layers", type=int, default=0, help="from-scratch: override #layers")
    ap.add_argument("--heads", type=int, default=0, help="from-scratch: override #attn heads")
    ap.add_argument("--out", default="models/neoperi-encoder")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)
    from datasets import Dataset

    # ModernBERT (the modern arch: RoPE + GeGLU + local/global attention) needs a
    # recent transformers. Fail fast with a clear message rather than KeyError.
    if args.from_scratch or "modernbert" in args.base.lower():
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.48.0"):
            sys.exit(f"ModernBERT/modern-arch needs transformers>=4.48 "
                     f"(have {transformers.__version__}); run: pip install -U 'transformers>=4.48'")

    print("=" * 70)
    print(f"neoperi ENCODER  base={args.base}  "
          f"mode={'FROM-SCRATCH (modern arch)' if args.from_scratch else 'domain-adaptive'}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.mask_token_id is None:
        sys.exit("ABORT: tokenizer has no [MASK] token; MLM needs one.")
    if args.from_scratch:
        cfg = AutoConfig.from_pretrained(args.base)   # architecture (+vocab) from base
        for attr, val in (("hidden_size", args.hidden), ("num_hidden_layers", args.layers),
                          ("num_attention_heads", args.heads)):
            if val and hasattr(cfg, attr):
                setattr(cfg, attr, val)
        model = AutoModelForMaskedLM.from_config(cfg)   # RANDOM weights, modern arch
        print(f"==> Random-init encoder: {getattr(cfg,'num_hidden_layers','?')}L "
              f"/ {getattr(cfg,'hidden_size','?')}h / {getattr(cfg,'num_attention_heads','?')} heads "
              f"({cfg.model_type}).")
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.base)
    if model.get_input_embeddings().num_embeddings < len(tok):
        model.resize_token_embeddings(len(tok))

    if args.selftest:
        texts = ["Yenidoğan sarılığında total serum bilirubin gebelik haftasına göre "
                 "yorumlanır. Prematüre bebeklerde fototerapi eşiği daha düşüktür."] * 64
    else:
        texts = load_texts(args.corpus, args.extra_hf)
    if len(texts) < 8:
        sys.exit(f"ABORT: only {len(texts)} texts — build a bigger corpus first "
                 "(build_corpus.py with sources=europepmc,pubmed,hfds).")
    print(f"==> {len(texts)} training passages.")

    # Domain-adaptive MLM: concatenate + group into fixed blocks so NO token is
    # wasted and every position contributes (BioBERTurk/TurkRadBERT recipe).
    def tok_fn(batch):
        return tok(batch["text"], return_special_tokens_mask=True)
    tokd = Dataset.from_dict({"text": texts}).map(
        tok_fn, batched=True, remove_columns=["text"], desc="tokenizing")
    block = args.max_len

    def group_texts(ex):
        concat = {k: sum(ex[k], []) for k in ex}
        total = (len(concat["input_ids"]) // block) * block
        if total == 0:
            return {k: [] for k in concat}
        return {k: [v[i:i + block] for i in range(0, total, block)] for k, v in concat.items()}
    ds = tokd.map(group_texts, batched=True, desc="grouping into blocks")
    print(f"==> {len(ds)} MLM blocks of {block} tokens.")
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mlm_prob)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=args.out, overwrite_output_dir=True,
        num_train_epochs=(0.02 if args.selftest else args.epochs),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.06,
        logging_steps=20, save_strategy="epoch", save_total_limit=1,
        bf16=bf16, fp16=not bf16, report_to="none", seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "PROVENANCE.json"), "w", encoding="utf-8") as fh:
        json.dump({"kind": "encoder", "base": args.base,
                   "from_scratch": args.from_scratch, "n_texts": len(texts),
                   "note": "Turkish neoperi encoder for retrieval/NER. Domain-adaptive "
                           "MLM (or from-scratch). NOT a card generator; NOT clinically "
                           "validated."}, fh, ensure_ascii=False, indent=2)
    print(f"==> Saved encoder to {args.out}. Use it to embed guideline passages / NER.")


if __name__ == "__main__":
    main()
