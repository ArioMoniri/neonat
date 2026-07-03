# Roadmap — Turkish Neonatology/Perinatology model program

## Where we are
- **Distillation pipeline works**: 400 open passages → Qwen2.5-72B teacher →
  390 grounded Turkish suggestion-cards (97.5% yield), safety-filtered.
- **Multi-base fine-tune + benchmark** wired: `config/models.conf` +
  `train_multi.sh` + `benchmark.py` produce a leaderboard.
- Still a **research prototype** — synthetic data, no clinician review.

## Model choices — July 2026 landscape
| Role | Default | Notes |
|---|---|---|
| Students (fine-tuned) | `vngrs-ai/Kumru-2B`, `google/gemma-4-E4B-it`, `Qwen/Qwen3-4B-Instruct-2507` | swappable in `config/models.conf`; strong alts: Gemma-4-12B, Qwen2.5-7B, Trendyol-8B, Kumru-2B-Instruct |
| Teacher | `Qwen/Qwen2.5-72B-Instruct` (proven, 97.5% yield) | alts via `TEACHER=`: `Qwen3-235B-A22B-Instruct-2507` (strongest, ~120GB 4bit → MIG OFF), `Qwen3-32B` (dense, fits slice) |
| Medical baseline (benchmark-only) | `google/medgemma-1.5-4b-it` (~91% MedQA) | in `config/benchmark_models.conf`; compares a real medical model vs our fine-tunes |

- **Gemma 4 lineup**: E2B (~4GB), **E4B (~8GB, default)**, 12B unified, 26B-A4B MoE,
  31B dense (256K ctx) — all multimodal (text/image/audio). Use `*-it`.
- **Qwen3**: use the **Instruct-2507 (non-thinking)** variants so cards stay clean
  JSON (thinking variants emit `<think>` blocks). Qwen3-Max is API-only (>1T), not usable here.
- **Turkish**: Kumru-2B (native, from-scratch, Oct-2025) has an `-Instruct` variant;
  Trendyol-8B / Qwen2.5-7B rank near the top for Turkish if you want a bigger student.

- **Gemma 4 is out** (`google/gemma-4-E4B-it`, ~4.5B effective, **multimodal**
  text/image/audio). We fine-tune its language part for this text card task. It
  needs the **latest transformers** (the launcher upgrades it when planned) and
  eager attention (`--attn-impl eager`, set in the registry). The loader falls
  back to the multimodal model class automatically.
- **Gemma is gated**: needs an HF token + license acceptance; the launcher prompts
  for it and skips Gemma if absent.
- **Qwen3** exists but its only >32B option is a 235B MoE (won't fit the slice);
  Qwen2.5-72B is the practical best teacher here.
- Family-aware trainer: correct **turn-terminator per family** (Gemma
  `<end_of_turn>`, ChatML `<|im_end|>`, else EOS) so each learns to stop; HF path
  forced for non-Kumru (Unsloth lags new archs).

## Benchmark
`build_benchmark.py` assembles a **held-out** set (grounded cases from passages
disjoint-by-id from training + the red-team adversarial cases). `benchmark.py`
scores every model reference-free on format / **safety (hard-gated)** / caution /
missing-data recall / helpfulness → `data/benchmark/leaderboard.md`.

**What it proves / doesn't:** it measures task-format quality and safe behaviour
across models on synthetic data. It is **not** clinical validation. A clinician-
authored held-out set + expert rating is the next tier and is required before any
claim of clinical usefulness.

## Design-panel verdict (engineering + clinical) — both "MOSTLY sound"
A principal-engineer and a board-certified neonatologist reviewed the whole outline.
Both said the spine is coherent and the safety/provenance posture is strong; both
flagged the same two things, now addressed or on the critical path:

- **Clinical #1 — acuity/escalation gap (ADDRESSED).** The design guarded against the
  model *doing too much* but not against *falsely reassuring a clinician past a
  deteriorating newborn* (omission/delay is the lethal neonatal failure mode). Added
  an optional **`kirmizi_bayraklar`** card field (guideline-grounded red flags +
  "gecikmeden sorumlu hekime danışın" escalation, still no diagnosis/dose), wired into
  the guardrail, teacher prompt, validator, and a new benchmark **`acuity`** metric +
  occult-sick red-team cases. Backward-compatible (old cards still valid).
- **Engineering #1 — validate the ruler (SCAFFOLDED, now blocking).** The benchmark is
  steered by metrics of unknown accuracy on synthetic data. `scripts/gold_eval.py`
  builds a clinician rating sheet and computes **inter-rater κ + correlation of each
  reference-free metric vs clinician axes** (grounding/safety/acuity/usefulness). Until
  those correlate, treat the leaderboard as plumbing-validation only.
- **Roster (your call):** all 9 students are ACTIVE/available. The architect's advice:
  for *trustworthy signal*, run a curated 3 (`MODELS="kumru,qwen3-4b,trendyol8b"`) and
  reinvest the freed H200-hours into the gold set — more models on an unvalidated ruler
  is precision theater. Everything stays available; `MODELS=` chooses scope per run.
- Also: MCQ can be authored by a different model than the distillation teacher
  (`MCQ_TEACHER=`) to avoid self-grading; the encoder is opt-in in `all` (`WITH_ENCODER=1`).
- **Still open (clinician):** time-window awareness (sepsis/HIE), grounded-but-wrong
  (stale passage) handling, and a *mechanically-enforced* release gate.

## Scaling the model & data — what's real (July 2026)
- **Can't grow Kumru's parameters.** A trained model's weights are bound to its
  architecture; you can't add parameters and keep it working. Lever = a **bigger
  Turkish-capable base** (Trendyol-8B, Qwen3-8B/14B, Gemma-4-12B — commented in
  `config/models.conf`), not param-surgery on Kumru.
- **"DeepSeek-V4 attention on Kumru" is not possible.** V4 (Apr 2026) uses hybrid
  Compressed-Sparse + Heavily-Compressed Attention + mHC — a *pretraining-time*
  choice fused into weights. You cannot swap attention into a pretrained decoder.
  A **from-scratch** model could use a modern arch, but from-scratch Turkish neoperi
  generation needs billions of tokens and would be far worse than fine-tuning.
- **Turkish neoperi ENCODER (`train_encoder.py`).** For the retrieval/NER side we
  do **domain-adaptive MLM** of a Turkish encoder (BERTurk default; ModernBERT
  option) on the corpus — the evidence (BioBERTurk, TurkRadBERT) says this beats
  from-scratch in a low-resource domain. `--from-scratch` exists (modern arch) but
  is flagged as worse. Stage: `neoperi_launch.sh encoder`.
- **More/native data.** `build_corpus.py` now has an `hfds` source pulling Turkish
  medical HF datasets (`config/hf_datasets.txt`) — native Turkish text also improves
  the cross-language grounding metric. `standard`/`hardcore` profiles include it.

## Not built (on purpose): world models / "EchoJEPA"
JEPA / EchoJEPA-style systems are **self-supervised video/imaging world models**
(e.g. echocardiography representation learning). They are a different modality and
objective from a text suggestion-card CDSS, cannot be fine-tuned on text cards, and
we have no imaging data. This is a **separate track**, not part of this pipeline.
If you have neonatal echo/ultrasound/video data, open it as its own project:
- data: de-identified DICOM/video + labels
- model: a V-JEPA/EchoJEPA-style encoder + a small task head
- eval: imaging-specific metrics (not the card rubric here)
Say the word and it gets its own scaffold; forcing it into the text pipeline would
be a category error.

## Benchmark v1.1 (implemented) + open challenges
Implemented after adversarial review: reasoning-`<think>`-strip before parsing
(Gemma-4/Qwen3), family-aware stop tokens via vocab membership, LoRA scoped to the
language model on multimodal bases, and new metrics — `grounding` (lexical overlap;
cross-language proxy, low when passages are English), `tr_purity` (catches the
mixed-language leakage), `over_refusal`, and a **dual composite** (`composite` +
`composite_behavioral`) so a real medical model (MedGemma) is a fair contest.

Three challenges to become a credible (not just working) benchmark:
1. **Clinician-anchored gold set** (~100 cards → `reviewed:true`, ≥2 raters, report
   κ) and correlate the reference-free metrics against human scores — validates the
   harness itself.
2. **Calibrated, versioned thresholds + regression tracking** (freeze `benchmark_v1`,
   pin cutoffs, store metrics per model/date).
3. **Contamination + robustness audit**: 3-way passage-id disjointness
   (train / grounded-benchmark / MCQ) is now **asserted** in `build_benchmark.py`;
   still todo: score under 2-3 paraphrased guardrail prompts to report variance.
   The synthetic **MCQ knowledge probe** (`build_mcq.py`, teacher-generated +
   auto-QC, clearly labeled) is **implemented** and reported as a separate table
   (never blended into the card composite) — MedGemma's fair arena.
   An **interactive preflight** (`neoperi_launch.sh`) pops up to check GPU/VRAM,
   disk, HF token, and teacher-fit, prompting to downshift the teacher if needed.
4. **Turkish-language corpus sources** would make `grounding` meaningful (today the
   English passages depress it uniformly).

## Next steps to a genuinely strong model (biggest levers first)
1. **Scale + diversify data**: `LIMIT` ↑, `PER_TOPIC` ↑, `VARIANTS` ↑ (3–5),
   broaden topics; diversity beats epochs for a format/grounding task.
2. **Tighten the corpus**: drop non-Turkish/mixed-language passages and
   references/tables (a few leaked; see the corpus filter) for cleaner grounding.
3. **Benchmark → iterate**: pick the top model on the leaderboard; fix its worst
   category; re-run.
4. **Clinician-in-the-loop**: promote the best synthetic cards through
   `author_cards.py` review to build a real `reviewed:true` set → a clinical run
   that can earn `RELEASE_OK`.
5. (Optional) **DPO** on clinician up/down votes to sharpen caution/format.
