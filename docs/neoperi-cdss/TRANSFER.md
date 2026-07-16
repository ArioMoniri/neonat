# Transfer & Run Runbook — Kumru-2B TR neoperi CDSS

End-to-end: build a bundle locally → copy to the GPU server → set up an isolated
venv → train → gate → tear down. **Everything on the server lives in one folder,
so deleting it all is `rm -rf ~/neoperi-cdss`.**

> Your server, from your earlier session: `root@10.6.110.10`, SSH port `30405`.
> Change these to match your box.

---

## 0. Zero-data "plug & train" (no clinician data, no API keys)

If you have **no clinician-reviewed data**, use the distillation pipeline: it pulls
open neonatology/perinatology literature, has a teacher LLM (default
`Qwen/Qwen2.5-72B-Instruct`, runs on your H200) generate grounded Turkish cards,
validates them, and fine-tunes Kumru-2B in **synthetic mode**.

```bash
# on the server, after setup_server.sh:
bash scripts/plug_and_train.sh --plumbing          # offline dry-run of the data path
bash scripts/plug_and_train.sh                     # the real thing (downloads the teacher)

# tune it via env vars:
SOURCES="europepmc,pubmed,urls" URLS_FILE=data/corpus/guideline_urls.txt \
LIMIT=600 TEACHER="Qwen/Qwen2.5-72B-Instruct" RUN=synth-01 \
  bash scripts/plug_and_train.sh
```

> ⚠️ **Research prototype only.** The model is trained on **machine-generated**
> data, so it is NOT clinician-reviewed and **NOT for clinical use**. It gets a
> `RESEARCH_GATE_OK` (never a clinical `RELEASE_OK`), and the adapter carries a
> `PROVENANCE.json` marking it synthetic. To make it clinical-grade, feed real
> clinician-reviewed data via `author_cards.py` and train without
> `--allow-synthetic` (steps 4–6 below).

The steps below (manual bundle → train → gate) are the underlying pieces; section
0 just chains them with auto-built data.

---

## 1. Build the bundle (LOCAL, in this repo)

```bash
bash scripts/make_bundle.sh
```

Produces `dist/neoperi-cdss-bundle.tar.gz` containing code, configs, docs, and the
`*.example.jsonl` files only. **No real/clinician data and no large binaries.**

## 2. Copy it to the server (LOCAL)

```bash
scp -P 30405 dist/neoperi-cdss-bundle.tar.gz root@10.6.110.10:~/
```

If you also have clinician-reviewed training data ready, copy it separately
(it is never in the bundle) into the same project once extracted:

```bash
# example — copy your reviewed data straight into the project's data dir
scp -P 30405 path/to/task_sft.jsonl root@10.6.110.10:~/neoperi-cdss/data/processed/
```

## 3. Set up on the server (SSH)

```bash
ssh -p 30405 root@10.6.110.10
tar xzf neoperi-cdss-bundle.tar.gz
cd neoperi-cdss
bash scripts/setup_server.sh
```

`setup_server.sh` creates `./.venv`, installs the pinned requirements
(`config/requirements-train.txt`), points the model
cache at `./.hf_cache` (so downloads stay inside the project), and writes a
footprint manifest to `state/MANIFEST.txt`. It is idempotent — safe to re-run.

## 4. Train (SSH, on the server)

```bash
# Plumbing smoke test first (~20 steps) — validates the data gate + masking:
bash scripts/run_train.sh data/processed/task_sft.jsonl --smoke-only

# Full run (adapter -> models/kumru-neoperi-lora-run-01):
bash scripts/run_train.sh data/processed/task_sft.jsonl run-01
```

The training script **hard-aborts** unless every row is `reviewed:true`,
schema-valid, and grounded — so this step only runs on clinician-approved data.

## 5. Gate / evaluate (SSH)

```bash
bash scripts/run_eval.sh --adapter models/kumru-neoperi-lora-run-01 \
    --redteam data/redteam/redteam.example.jsonl
```

Writes `metrics.json` and, only if every **critical** safety check passes, a
`RELEASE_OK` file into the adapter dir (otherwise `RELEASE_BLOCKED`). This is the
mechanical go/no-go gate. **It is necessary, not sufficient — a neonatologist +
perinatologist must still sign off before any real-world use.**

## 6. Authoring more data (optional, anywhere with Python)

```bash
python scripts/author_cards.py template --out data/staging/passages.jsonl
# curate real guideline passages into that file, then:
python scripts/author_cards.py build --passages data/staging/passages.jsonl \
                                     --out data/staging/drafts.jsonl
# clinicians fill suggestions + set reviewed:true, then:
python scripts/author_cards.py promote --in data/staging/drafts.jsonl \
                                       --out data/processed/task_sft.jsonl
```

---

## 7. What gets created on the server (footprint)

All inside `~/neoperi-cdss/`:

| Path | What |
|------|------|
| `.venv/` | Python venv + all pip packages |
| `.hf_cache/` | Hugging Face model cache (Kumru-2B downloads here) |
| `models/` | trained LoRA adapters (+ `metrics.json`, `RELEASE_OK`) |
| `state/` | `install.log`, `pip-freeze.txt`, `torch-check.txt`, `MANIFEST.txt` |
| `env.sh` | generated env (HF_HOME etc.) |
| `data/processed/task_sft.jsonl` | your reviewed data (you put it here) |

## 8. Delete everything (SSH)

```bash
# Remove generated footprint (venv, model cache, adapters) but keep source + data:
bash scripts/uninstall.sh

# Or nuke the entire project, including data:
bash scripts/uninstall.sh --all        # asks for confirmation
# Simplest equivalent:
rm -rf ~/neoperi-cdss
rm -f  ~/neoperi-cdss-bundle.tar.gz
```

Because the venv and the model cache both live inside the project folder, a single
`rm -rf ~/neoperi-cdss` reclaims **all** disk this project used — no stray caches
in `~/.cache/huggingface` or system site-packages.
