# Kumru-2B — Turkish Neonatology / Perinatology CDSS fine-tune

A parameter-efficient (QLoRA) fine-tune of **`vngrs-ai/Kumru-2B`** (Apache-2.0,
Mistral architecture) that teaches the model the **Turkish suggestion-card format
and clinical phrasing** for a neonatology + perinatology (perinatoloji) clinical
decision-support assistant.

> **What this fine-tune does NOT do:** it does not teach medical *facts*. Clinical
> knowledge is supplied by **retrieval at inference time**. Training only shapes
> task behaviour, output schema, register, and caution.

---

## 1. The single file to transfer

Everything needed to train is in one self-contained script:

```
scripts/train_lora.py
```

`scp` it (plus your reviewed data) to the GPU box and run it. It bootstraps its own
Python dependencies, validates the data, trains a 4-bit QLoRA adapter, and runs a
base-vs-tuned sanity check.

```bash
# on the H200 host
python train_lora.py --install-deps                         # one-time
python train_lora.py data/processed/task_sft.jsonl my-run-01
```

Smoke test only (plumbing, ~20 steps, no full run):

```bash
python train_lora.py data/processed/task_sft.jsonl --smoke-only
```

Force the plain Hugging Face path (skip Unsloth):

```bash
python train_lora.py data/processed/task_sft.jsonl --no-unsloth
```

Output adapter (LoRA + tokenizer) lands in `models/kumru-neoperi-lora[-<run>]/`.

---

## 2. Data contract

One JSON object per line in `data/processed/task_sft.jsonl`
(see `data/processed/task_sft.example.jsonl` for the exact shape):

```json
{"messages": [
  {"role": "system",    "content": "<CDSS task instruction + guardrails>"},
  {"role": "user",      "content": "<patient context + retrieved guideline passage>"},
  {"role": "assistant", "content": "<suggestion-card JSON>"}
 ],
 "reviewed": true}
```

The suggestion-card (assistant content) is valid JSON with these keys:

| key                  | meaning                                              |
|----------------------|------------------------------------------------------|
| `onerilen_sorular`   | suggested questions to ask                           |
| `onerilen_tetkikler` | suggested tests/work-up to consider                  |
| `eksik_veriler`      | clinically pivotal data that is missing              |
| `kaynak`             | the grounding guideline passage id (must be real)    |
| `uyari`              | caution / not-a-substitute-for-clinician disclaimer  |

**Fail-closed gate.** A row reaches the weights only if it is *provably*
`reviewed:true` **and** schema-valid **and** grounding-consistent. The script
**hard-aborts** (never a warning) on the first violation across all rows:

- `reviewed:true` missing → abort.
- Unparseable line, missing user/assistant turn, empty user passage → abort.
- Assistant target not valid card JSON, or missing any of the 5 keys, or `uyari`
  empty → abort.
- **Grounding invariant:** if the card contains any suggestion
  (`onerilen_sorular` or `onerilen_tetkikler`), `kaynak` **must** be non-empty.
  A card with no citation is allowed *only* as the safe "everything is missing"
  card (empty suggestions, populated `eksik_veriler`) — see the 2nd example row.

A non-fatal **red-flag** scan also warns when card text matches decision/dose
patterns (e.g. `mg/kg`, `doz:`, `reçete`) so `cdss-safety-redteam` can confirm the
row only *suggests*. "It didn't abort" therefore means every row is approved,
well-formed, and grounded — no silent exceptions.

---

## 3. Correctness guarantees baked into the script

- **Assistant-only loss masking, by construction** — the prompt is rendered with the
  generation prompt, then the assistant content + a **real EOS** is appended as the
  supervised span; the prompt is masked to `-100`. No fragile two-render diffing, and
  every target ends on a real stop token (the model learns to stop). Per-example
  supervised-token telemetry is printed as a guard against silent mask corruption.
- **Pad token handled** — a pad token *distinct from EOS* is set (asserts
  `pad != eos`); label padding is `-100`. Embeddings are resized on **both** the
  Unsloth and HF paths if a pad token was added.
- **Chat template** — uses Kumru's own template; falls back to ChatML if absent.
- **GPU-adaptive** — reads `nvidia-smi` (incl. MIG) and lowers batch / raises accum if
  memory is tight. On an H200 MIG slice, a 2B 4-bit model fits with huge headroom.
- **Smoke → full** — runs ~20 steps first and checks the loss is finite before the
  full run.

---

## 4. The review/quality team (`.claude/agents/`)

A finished run is **"ready to evaluate", never "ready to use."** These subagents carry
the work from data to a gated release:

| Agent                      | Role |
|----------------------------|------|
| `clinical-data-curator`    | Builds/validates the jsonl, enforces schema + `reviewed:true`, routes rows |
| `neonatologist-reviewer`   | Clinical sign-off for newborn/NICU content |
| `perinatologist-reviewer`  | Clinical sign-off for pregnancy/fetal/peripartum content |
| `turkish-medical-linguist` | Native Turkish terminology, register, consistency |
| `ml-finetune-engineer`     | Owns `train_lora.py`, the recipe, masking, GPU fit |
| `cdss-safety-redteam`      | Adversarial safety set; blocks releases that diagnose/order/hallucinate |
| `eval-benchmark-lead`      | Held-out metrics + go/no-go report |

Typical loop: **curate → clinical review → linguist → train → red-team + eval → fix → repeat.**

---

## 5. Before any clinical use

Decreasing training loss ≠ safe cards. Required before use:

1. Held-out clinician case evaluation (`eval-benchmark-lead`).
2. Missing-data / citation-grounding **red-team** must pass critical cases
   (`cdss-safety-redteam`).
3. Format + scope + caution metrics above threshold.

The gate above is **procedural** (personas + this doc). To make it *mechanical*,
have `eval-benchmark-lead` + `cdss-safety-redteam` emit a signed `RELEASE_OK` file
into the adapter dir only when critical thresholds pass, and have any serving code
refuse to load an adapter without it.

Optional later: a small DPO pass on clinician up/down votes to tune format and caution.
Keep adapters small and per-task.
