# Roadmap — Turkish Neonatology/Perinatology model program

## Where we are
- **Distillation pipeline works**: 400 open passages → Qwen2.5-72B teacher →
  390 grounded Turkish suggestion-cards (97.5% yield), safety-filtered.
- **Multi-base fine-tune + benchmark** wired: `config/models.conf` +
  `train_multi.sh` + `benchmark.py` produce a leaderboard.
- Still a **research prototype** — synthetic data, no clinician review.

## Model choices (and the naming reality)
| Role | Default | Notes |
|---|---|---|
| Students (fine-tuned) | `vngrs-ai/Kumru-2B`, `google/gemma-4-E4B-it`, `Qwen/Qwen2.5-3B-Instruct` | all swappable in `config/models.conf` |
| Teacher | `Qwen/Qwen2.5-72B-Instruct` | swappable via `TEACHER=` |

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
