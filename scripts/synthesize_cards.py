#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_cards.py — distill grounded Turkish suggestion-cards with a teacher LLM.
================================================================================
For each open-literature passage, a strong open TEACHER model (default
Qwen2.5-72B-Instruct, 4-bit on the H200) generates:
  • a short SYNTHETIC Turkish neonatal/perinatal patient vignette, and
  • a suggestion-card grounded ONLY in that passage (questions + tests to consider,
    missing data, kaynak, caution) — never a diagnosis/dose/order.

Each generated card is validated with the SAME validator training uses
(train_lora.validate_card). Invalid/ungrounded generations are discarded. Output
rows are written as SYNTHETIC (reviewed:false + provenance.source=="auto"), so
train_lora must be run with --allow-synthetic and the result is a research
prototype — NOT clinician-reviewed, NOT for clinical use.

Usage:
    python synthesize_cards.py --passages data/corpus/passages.jsonl \
        --out data/processed/task_sft.synth.jsonl \
        --teacher Qwen/Qwen2.5-72B-Instruct --limit 400
    python synthesize_cards.py --passages ... --out ... --dry-run   # stub teacher
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_train_module():
    path = os.path.join(_HERE, "train_lora.py")
    spec = importlib.util.spec_from_file_location("train_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TL = _load_train_module()
GUARDRAIL_SYSTEM = TL.GUARDRAIL_SYSTEM
DEFAULT_UYARI = ("Bu öneriler yalnızca verilen kılavuza dayanır ve klinik karar "
                 "yerine geçmez; nihai değerlendirme hekime aittir.")

TEACHER_SYSTEM = (
    "Sen, Türkçe konuşan bir tıbbi veri etiketleme uzmanısın. Görevin, verilen "
    "İNGİLİZCE/Türkçe kılavuz pasajından, neonatoloji/perinatoloji için bir klinik "
    "karar destek 'öneri kartı' eğitim örneği üretmektir. "
    "SADECE geçerli JSON üret, başka hiçbir şey yazma. Kurallar: "
    "(1) Önce pasajla ilgili KISA, sentetik (gerçek olmayan) bir Türkçe hasta "
    "olgu senaryosu (vignette) yaz. "
    "(2) Öneriler SADECE pasajdaki bilgiye dayanmalı. "
    "(3) Asla tanı koyma, ilaç/doz/order yazma; sadece SORULACAK sorular ve "
    "DEĞERLENDİRİLECEK tetkikler öner. "
    "(4) Pasajda olmayan kritik verileri 'eksik_veriler' altında belirt. "
    "(5) 'kaynak' alanına sana verilen passage_id'yi aynen yaz. "
    "(6) Vinyette/pasajda gecikmeye tahammülü olmayan acil bulgular (letarji, "
    "kötü perfüzyon, apne, konvülziyon, safralı kusma, siyanoz) varsa "
    "'kirmizi_bayraklar' altında yaz ve gecikmeden sorumlu hekime danışılmasını "
    "öner (yine de tanı/doz verme); yoksa boş liste bırak. "
    "Çıktı şeması: {\"vignette\":\"...\",\"onerilen_sorular\":[],"
    "\"onerilen_tetkikler\":[],\"eksik_veriler\":[],\"kaynak\":\"<passage_id>\","
    "\"uyari\":\"...\",\"kirmizi_bayraklar\":[]}"
)


def build_teacher_prompt(passage, passage_id):
    return (f"passage_id: {passage_id}\n\nKılavuz pasajı:\n\"\"\"\n{passage}\n\"\"\"\n\n"
            f"Yukarıdaki şemada, SADECE JSON olarak bir öneri kartı üret.")


# ----------------------------------------------------------------------------
def extract_json(text):
    """Pull the first balanced {...} object out of a model response. If the teacher
    output was truncated (unbalanced), attempt a salvage by closing the open string
    and braces — recovers many otherwise-dropped generations."""
    start = text.find("{")
    if start < 0:
        return None
    stack, in_str, esc = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()
                if not stack:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # Salvage a truncated object/array: close the dangling string, drop a trailing
    # partial token, then close every open bracket/brace in reverse order.
    tail = text[start:]
    if in_str:
        tail += '"'
    if stack:
        tail = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", tail)  # partial "key":
        tail = re.sub(r',\s*$', "", tail)
        tail += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        return None


def to_card(obj, passage_id):
    """Coerce a teacher object into a clean 5-key card grounded on passage_id."""
    if not isinstance(obj, dict):
        return None
    card = {
        "onerilen_sorular": obj.get("onerilen_sorular", []),
        "onerilen_tetkikler": obj.get("onerilen_tetkikler", []),
        "eksik_veriler": obj.get("eksik_veriler", []),
        "kaynak": passage_id,                       # force correct grounding
        "uyari": (obj.get("uyari") or DEFAULT_UYARI),
        "kirmizi_bayraklar": obj.get("kirmizi_bayraklar", []),   # acuity/escalation
    }
    for k in ("onerilen_sorular", "onerilen_tetkikler", "eksik_veriler", "kirmizi_bayraklar"):
        v = card[k]
        if isinstance(v, str):
            v = [v]
        card[k] = [str(x).strip() for x in v if str(x).strip()][:8]
    return card


def build_row(card, vignette, passage, src):
    user = (f"Hasta bağlamı: {vignette.strip()}. "
            f"Kılavuz pasajı: {passage.strip()}")
    return {
        "messages": [
            {"role": "system", "content": GUARDRAIL_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(card, ensure_ascii=False)},
        ],
        "reviewed": False,
        "provenance": {
            "source": "auto", "teacher": src.get("teacher"),
            "passage_id": src.get("passage_id"), "passage_source": src.get("source"),
            "license": src.get("license"), "url": src.get("url"),
            "variant": src.get("variant", 0), "category": src.get("category", "grounded"),
        },
    }


# ---------------------------------------------------------------------------
# REFUSAL partition: the fix for the empty_passage/boundary/acuity collapse.
# The teacher authors a first-class "cannot ground this" card (karar='refusal',
# kaynak=null) so the student learns to abstain in valid-card form instead of
# fabricating a citation. Blended ~1:3 refusal:grounded (--refusal-ratio).
# ---------------------------------------------------------------------------
ESCALATION_UYARI = ("Gecikmeden acil hekim/yenidoğan değerlendirmesi gereklidir; "
                    "bu kart klinik karar yerine geçmez.")
REFUSAL_SCENARIOS = ("empty", "boundary", "acuity")
REFUSAL_TEACHER_SYSTEM = (
    "Sen Türkçe tıbbi veri etiketleme uzmanısın. Görevin, modelin GÜVENLE "
    "TEMELLENDİREMEYECEĞİ bir durum için geçerli bir 'RET kartı' (refusal card) "
    "üretmek. SADECE geçerli JSON üret. Sana bir senaryo tipi verilecek: "
    "(empty) yeterli/ilgili kılavuz pasajı yok; (boundary) kullanıcı tanı/doz/"
    "kesin karar dayatıyor ama verilmemeli; (acuity) vinyette gecikmeye tahammülü "
    "olmayan acil bulgular var. Kurallar: 'karar' değeri 'refusal' olmalı; ASLA "
    "tanı koyma, ilaç/doz/order verme; 'onerilen_tetkikler' BOŞ liste olmalı; "
    "'eksik_veriler' neden temellendirilemediğini ve hangi kritik verinin "
    "gerektiğini listelemeli (boş olamaz); 'gerekce' kısa Türkçe açıklama olmalı; "
    "'kaynak' null olmalı; acil bulgu varsa 'kirmizi_bayraklar' doldur ve 'uyari' "
    "GECİKMEDEN acil hekim/yenidoğan değerlendirmesi istemeli (ör. 'gecikmeden acil "
    "hekim değerlendirmesi gereklidir'). Şema: {\"karar\":\"refusal\",\"vignette\":"
    "\"...\",\"onerilen_sorular\":[],\"onerilen_tetkikler\":[],\"eksik_veriler\":"
    "[...],\"gerekce\":\"...\",\"kirmizi_bayraklar\":[],\"kaynak\":null,\"uyari\":\"...\"}")


def build_refusal_prompt(scenario, passage):
    kind = {
        "empty": ("Senaryo tipi: empty. İlgili/yeterli kılavuz pasajı YOK. Aşağıdaki "
                  "metin ya boş ya da soruyla ilgisiz:"),
        "boundary": ("Senaryo tipi: boundary. Kullanıcı kesin tanı veya ilaç/doz "
                     "istiyor; bu verilmemeli. Pasaj kesin kararı desteklemiyor:"),
        "acuity": ("Senaryo tipi: acuity. Vinyette gecikmeye tahammülü olmayan acil "
                   "bulgular var; öncelik acil hekim değerlendirmesine yönlendirmek:"),
    }[scenario]
    body = (passage or "(boş)").strip()
    return (f"{kind}\n\"\"\"\n{body[:1500]}\n\"\"\"\n\n"
            f"Yukarıdaki şemada SADECE JSON olarak bir RET kartı üret.")


def teacher_generate_refusal(model, tok, scenario, passage, max_new_tokens=512):
    import torch
    msgs = [{"role": "system", "content": REFUSAL_TEACHER_SYSTEM},
            {"role": "user", "content": build_refusal_prompt(scenario, passage)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    dev = model.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def to_refusal_card(obj, scenario):
    """Coerce a teacher object into a schema-valid refusal card (kaynak=null)."""
    if not isinstance(obj, dict):
        return None
    def _lst(key, cap=8):
        v = obj.get(key, [])
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in (v or []) if str(x).strip()][:cap]
    eksik = _lst("eksik_veriler") or ["Güvenli temellendirme için yeterli/ilgili kılavuz verisi yok."]
    kirmizi = _lst("kirmizi_bayraklar")
    gerekce = str(obj.get("gerekce") or "").strip() or \
        "Verilen bilgi güvenli, kılavuza dayalı bir öneri için yetersiz."
    uyari = str(obj.get("uyari") or "").strip()
    # Acuity refusals MUST carry urgent escalation language (validator enforces it).
    if kirmizi and not TL.ESCALATION_REGEX.search(uyari):
        uyari = ESCALATION_UYARI
    if not uyari:
        uyari = ("Bu durumda güvenli öneri verilemez; eksik veriler tamamlanmalı ve "
                 "hekime danışılmalıdır.")
    return {
        "karar": "refusal",
        "onerilen_sorular": _lst("onerilen_sorular"),
        "onerilen_tetkikler": [],                       # refusal -> empty by contract
        "eksik_veriler": eksik,
        "kaynak": None,                                 # ungroundable -> no citation
        "uyari": uyari,
        "kirmizi_bayraklar": kirmizi,
        "gerekce": gerekce,
    }


# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# AGENTIC exemplars: grounded cards that also carry STATE -> ACTION -> RESULT, with
# a read-only action verb and non-decisional text. validate_card enforces the verb
# whitelist + violates_action_policy, so any ordering/decisional output is dropped.
# ---------------------------------------------------------------------------
AGENTIC_TEACHER_SYSTEM = (
    "Sen Türkçe tıbbi veri etiketleme uzmanısın. Pasaja dayalı bir 'öneri kartı' üret "
    "AMA ek olarak DURUM-EYLEM-SONUÇ farkındalığı ekle. SADECE geçerli JSON. Kurallar: "
    "(1) Öneriler SADECE pasaja dayalı; 'kaynak' alanına verilen passage_id'yi yaz. "
    "(2) 'hasta_durumu': {gebelik_haftasi (sayı), postnatal_gun (sayı), aktif_problem "
    "(kısa), onceki_eylemler: [{eylem, zaman}]} — kısa ve sentetik. "
    "(3) 'onerilen_eylem': {verb: SADECE şu listeden [sor, gozlemle, "
    "tetkik_iste_degerlendirme_icin, hekime_danis]; aciklama: EMİR/DOZ/ORDER İÇERMEYEN "
    "kısa açıklama}. (4) 'eylem_sonucu': {eylem, gozlenen_sonuc (ör. lab değeri 'TSB 18 "
    "mg/dL'), durum_guncellemesi: KARAR/EŞİK/EMİR İÇERMEYEN, ör. 'hekimle "
    "değerlendirilmeli'}. (5) 'guven': 0-1 arası sayı. (6) 'gerekce': kısa açıklama. "
    "ASLA tanı koyma, ilaç/doz/order verme. Şema: {\"vignette\":\"...\","
    "\"onerilen_sorular\":[],\"onerilen_tetkikler\":[],\"eksik_veriler\":[],"
    "\"kaynak\":\"<passage_id>\",\"uyari\":\"...\",\"kirmizi_bayraklar\":[],"
    "\"hasta_durumu\":{\"gebelik_haftasi\":38,\"postnatal_gun\":2,\"aktif_problem\":"
    "\"...\",\"onceki_eylemler\":[]},\"onerilen_eylem\":{\"verb\":\"hekime_danis\","
    "\"aciklama\":\"...\"},\"eylem_sonucu\":{\"eylem\":\"...\",\"gozlenen_sonuc\":\"...\","
    "\"durum_guncellemesi\":\"...\"},\"guven\":0.7,\"gerekce\":\"...\"}")


def teacher_generate_agentic(model, tok, passage, passage_id, max_new_tokens=640):
    import torch
    msgs = [{"role": "system", "content": AGENTIC_TEACHER_SYSTEM},
            {"role": "user", "content": build_teacher_prompt(passage, passage_id)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    dev = model.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def to_card_agentic(obj, passage_id):
    """Grounded card + additive state/action/result. Sanitizes the verb to the read-only
    whitelist; violates_action_policy (in validate_card) drops any decisional free text."""
    card = to_card(obj, passage_id)
    if card is None:
        return None
    hd = obj.get("hasta_durumu")
    if isinstance(hd, dict):
        card["hasta_durumu"] = hd
    ae = obj.get("onerilen_eylem")
    if isinstance(ae, dict):
        verb = ae.get("verb")
        if verb not in TL.ACTION_VERBS:
            verb = "hekime_danis"                      # coerce to a safe read-only verb
        card["onerilen_eylem"] = {"verb": verb, "aciklama": str(ae.get("aciklama", "")).strip()}
    es = obj.get("eylem_sonucu")
    if isinstance(es, dict):
        card["eylem_sonucu"] = {
            "eylem": str(es.get("eylem", "")).strip(),
            "gozlenen_sonuc": str(es.get("gozlenen_sonuc", "")).strip(),
            "durum_guncellemesi": str(es.get("durum_guncellemesi", "")).strip()}
    g = obj.get("guven")
    if isinstance(g, (int, float)) and not isinstance(g, bool):
        card["guven"] = max(0.0, min(1.0, float(g)))
    gk = obj.get("gerekce")
    if isinstance(gk, str) and gk.strip():
        card["gerekce"] = gk.strip()
    return card


def load_teacher(model_id):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    print(f"==> Loading teacher {model_id} in 4-bit (first run downloads weights)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", **TL.hf_dtype_kwargs())
    model.eval()
    return model, tok


def teacher_generate(model, tok, passage, passage_id, max_new_tokens=512):
    import torch
    msgs = [{"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": build_teacher_prompt(passage, passage_id)}]
    # return_dict=True gives input_ids AND attention_mask (more correct than a bare
    # tensor, avoids pad/mask warnings). Place every tensor on the model's device.
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    dev = model.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Distill grounded TR cards with a teacher LLM.")
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", default="data/processed/task_sft.synth.jsonl")
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B",
                    help="distillation teacher (launcher passes Qwen3-235B primary / "
                         "Qwen3-32B fallback). apache-2.0 teachers give clean distill rights")
    ap.add_argument("--limit", type=int, default=400, help="max passages to process")
    ap.add_argument("--variants", type=int, default=1,
                    help="cards to generate PER passage (multiplies dataset size; "
                         "temperature sampling makes them diverse). e.g. 3 -> ~3x data")
    ap.add_argument("--refusal-ratio", type=float, default=0.25,
                    help="fraction of the final set that is REFUSAL cards (empty/"
                         "boundary/acuity) — teaches valid abstention. 0.25 -> ~1:3 "
                         "refusal:grounded. Set 0 to disable.")
    ap.add_argument("--agentic-ratio", type=float, default=0.15,
                    help="fraction of GROUNDED cards that also carry state->action->"
                         "result (hasta_durumu/onerilen_eylem/eylem_sonucu). 0 to disable.")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="raise if cards get truncated (vignette+lists can be long)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a stub teacher (no model/GPU) to test the pipeline")
    ap.add_argument("--append", action="store_true",
                    help="GROW an existing --out file (dedup) instead of overwriting")
    args = ap.parse_args()

    if not os.path.exists(args.passages):
        sys.exit(f"ABORT: passages file not found: {args.passages} (run build_corpus.py)")
    passages = [json.loads(l) for l in open(args.passages, encoding="utf-8") if l.strip()]
    passages = passages[:args.limit]
    if not passages:
        sys.exit("ABORT: no passages to process.")
    print(f"==> Synthesizing cards for {len(passages)} passage(s) with {args.teacher}")

    if args.dry_run:
        def gen(passage, pid):
            return json.dumps({
                "vignette": "Sentetik vinyet: term yenidoğan, postnatal 2. gün.",
                "onerilen_sorular": ["Postnatal kaçıncı gün?", "Beslenme öyküsü nasıl?"],
                "onerilen_tetkikler": ["Total serum bilirubin"],
                "eksik_veriler": ["gebelik haftası"],
                "kaynak": pid, "uyari": DEFAULT_UYARI}, ensure_ascii=False)

        def gen_ref(scenario, passage):
            return json.dumps({
                "karar": "refusal",
                "vignette": "Sentetik vinyet: bilgi yetersiz.",
                "onerilen_sorular": [], "onerilen_tetkikler": [],
                "eksik_veriler": ["gebelik haftası", "muayene bulguları"],
                "gerekce": "Yeterli/ilgili kılavuz verisi yok.",
                "kirmizi_bayraklar": (["letarji"] if scenario == "acuity" else []),
                "kaynak": None,
                "uyari": (ESCALATION_UYARI if scenario == "acuity"
                          else "Eksik veriler tamamlanmalı; hekime danışılmalıdır.")},
                ensure_ascii=False)

        def gen_ag(passage, pid):
            return json.dumps({
                "vignette": "Sentetik vinyet: term yenidoğan, postnatal 2. gün.",
                "onerilen_sorular": ["Beslenme öyküsü nasıl?"],
                "onerilen_tetkikler": ["Total serum bilirubin"],
                "eksik_veriler": ["gebelik haftası"], "kaynak": pid, "uyari": DEFAULT_UYARI,
                "kirmizi_bayraklar": [],
                "hasta_durumu": {"gebelik_haftasi": 38, "postnatal_gun": 2,
                                 "aktif_problem": "sarılık", "onceki_eylemler": []},
                "onerilen_eylem": {"verb": "hekime_danis", "aciklama": "hekime danışılmalı"},
                "eylem_sonucu": {"eylem": "TSB istendi", "gozlenen_sonuc": "18 mg/dL",
                                 "durum_guncellemesi": "fototerapi eşiği hekimle değerlendirilmeli"},
                "guven": 0.7, "gerekce": "pasaja dayalı"}, ensure_ascii=False)
        teacher_name = f"{args.teacher} (dry-run stub)"
    else:
        model, tok = load_teacher(args.teacher)
        gen = lambda passage, pid: teacher_generate(  # noqa: E731
            model, tok, passage, pid, args.max_new_tokens)
        gen_ref = lambda scenario, passage: teacher_generate_refusal(  # noqa: E731
            model, tok, scenario, passage, min(args.max_new_tokens, 640))
        gen_ag = lambda passage, pid: teacher_generate_agentic(  # noqa: E731
            model, tok, passage, pid, min(args.max_new_tokens, 768))
        teacher_name = args.teacher
        # PREFLIGHT: generate ONE card and surface the REAL error/traceback before
        # grinding the whole corpus. A failure here shows exactly what's wrong.
        print("==> Preflight: generating one card to validate the teacher...")
        try:
            probe = gen(passages[0]["passage"], passages[0].get("passage_id", "p1"))
            print(f"    preflight OK ({len(probe)} chars). First 200: {probe[:200]!r}")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(
                "\nABORT: teacher generation failed on the first passage (real "
                "traceback above).\nMost common causes & fixes:\n"
                "  • GPU/VRAM: the 72B may not fit your MIG slice alongside the KV "
                "cache. Try a smaller teacher:\n"
                "      TEACHER=Qwen/Qwen2.5-32B-Instruct bash scripts/plug_and_train.sh\n"
                "    or lower the token budget: --max-new-tokens 512\n"
                "  • If it's a device/offload error ('expected all tensors on the same "
                "device'), the model partially offloaded to CPU — use a smaller teacher.\n"
                "  • Re-running is cheap: the corpus and teacher weights are already "
                "cached; just re-run synthesize_cards.py on the existing passages file.")

    variants = max(1, args.variants)
    total = len(passages) * variants
    # Every Nth grounded generation carries state->action->result (agentic exemplar).
    agentic_period = round(1 / args.agentic_ratio) if args.agentic_ratio > 0 else 0
    print(f"==> {len(passages)} passage(s) x {variants} variant(s) = up to {total} cards"
          + (f"  (~1/{agentic_period} agentic)" if agentic_period else ""))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept, dropped = 0, 0
    drops = {"gen_error": 0, "no_json": 0, "invalid_card": 0, "prescribe": 0, "dup": 0}
    # --append GROWS an existing set (dedup against what's already there) instead
    # of overwriting — so you never lose already-generated cards.
    mode, seen_global = "w", set()
    if args.append and os.path.exists(args.out):
        mode = "a"
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                seen_global.add(line[:400])
        print(f"==> APPEND mode: {len(seen_global)} existing card(s) kept.")
    with open(args.out, mode, encoding="utf-8") as out_fh:
        for i, p in enumerate(passages, 1):
            pid = p.get("passage_id", f"p{i}")
            seen_cards = set()                    # dedup identical variants per passage
            for v in range(variants):
                ag = bool(agentic_period) and ((i + v) % agentic_period == 0)
                try:
                    raw = (gen_ag if ag else gen)(p["passage"], pid)
                except Exception as e:  # noqa: BLE001
                    drops["gen_error"] += 1
                    dropped += 1
                    # Real error (type + repr); full traceback on the first one —
                    # str(e) can be empty for some exception types.
                    print(f"    [{i}/{len(passages)} v{v}] {pid}: generation error: "
                          f"{type(e).__name__}: {e!r}")
                    if drops["gen_error"] == 1:
                        traceback.print_exc()
                    if drops["gen_error"] >= 5 and kept == 0:
                        sys.exit(
                            "ABORT: teacher generation failed on the first 5 attempts "
                            "(traceback above). Systemic issue, not bad data. Try a "
                            "smaller teacher (TEACHER=Qwen/Qwen2.5-32B-Instruct) or "
                            "--max-new-tokens 512. Corpus + weights are cached.")
                    continue
                obj = extract_json(raw)
                card = ((to_card_agentic(obj, pid) if ag else to_card(obj, pid))
                        if obj else None)
                if card is None:
                    drops["no_json"] += 1
                    dropped += 1
                    continue
                ok, reason = TL.validate_card(card)
                if not ok:
                    drops["invalid_card"] += 1
                    dropped += 1
                    continue
                blob = json.dumps(card, ensure_ascii=False)
                # Reject actual PRESCRIPTIONS (imperative order/dose) — but keep cards
                # that merely mention a guideline threshold/dose inside a question or
                # test. (Dropping on the broad looks_like_decision rejected ~all cards.)
                if TL.looks_like_prescription(blob):
                    drops["prescribe"] += 1
                    dropped += 1
                    continue
                if blob in seen_cards:            # identical variant — skip
                    drops["dup"] += 1
                    dropped += 1
                    continue
                seen_cards.add(blob)
                vignette = (obj.get("vignette") or "Sentetik hasta bağlamı")
                row = build_row(card, vignette, p["passage"],
                                {**p, "teacher": teacher_name, "variant": v,
                                 "category": ("agentic" if ag else "grounded")})
                line = json.dumps(row, ensure_ascii=False)
                if line[:400] in seen_global:     # already in the appended file
                    drops["dup"] += 1
                    dropped += 1
                    continue
                seen_global.add(line[:400])
                out_fh.write(line + "\n")
                kept += 1
            if i % 25 == 0 or i == len(passages):
                print(f"    [{i}/{len(passages)}] kept={kept} dropped={dropped} {drops}")

        # ---- REFUSAL partition (empty/boundary/acuity abstention exemplars) ----
        rr = max(0.0, min(0.9, args.refusal_ratio))
        refusal_target = int(round(kept * rr / (1 - rr))) if rr > 0 and kept > 0 else 0
        if refusal_target:
            import random as _random
            _random.seed(0)
            print(f"==> Generating ~{refusal_target} REFUSAL card(s) "
                  f"(ratio {rr:.2f} -> ~1:{max(1, round((1 - rr) / rr))} refusal:grounded)")
            r_made, attempts = 0, 0
            while r_made < refusal_target and attempts < refusal_target * 6 + 20:
                attempts += 1
                scenario = REFUSAL_SCENARIOS[attempts % len(REFUSAL_SCENARIOS)]
                src_p = passages[_random.randrange(len(passages))]
                passage_txt = "" if scenario == "empty" else src_p.get("passage", "")
                try:
                    raw = gen_ref(scenario, passage_txt)
                except Exception:  # noqa: BLE001
                    drops["refusal_gen"] = drops.get("refusal_gen", 0) + 1
                    if drops["refusal_gen"] == 1:
                        traceback.print_exc()
                    continue
                obj = extract_json(raw)
                card = to_refusal_card(obj, scenario) if obj else None
                if card is None:
                    drops["refusal_invalid"] = drops.get("refusal_invalid", 0) + 1
                    continue
                ok, _reason = TL.validate_card(card)
                if not ok:
                    drops["refusal_invalid"] = drops.get("refusal_invalid", 0) + 1
                    continue
                blob = json.dumps(card, ensure_ascii=False)
                if TL.looks_like_prescription(blob):
                    drops["prescribe"] += 1
                    continue
                cat = ("acuity" if scenario == "acuity"
                       else "empty_passage" if scenario == "empty" else "boundary_pressure")
                vignette = (obj.get("vignette") or "Sentetik hasta bağlamı (temellendirilemez)")
                row = build_row(card, vignette,
                                (passage_txt or "(ilgili/yeterli kılavuz pasajı yok)"),
                                {**src_p, "teacher": teacher_name, "variant": 0, "category": cat})
                line = json.dumps(row, ensure_ascii=False)
                if line[:400] in seen_global:
                    drops["dup"] += 1
                    continue
                seen_global.add(line[:400])
                out_fh.write(line + "\n")
                kept += 1
                r_made += 1
            print(f"==> refusal cards written: {r_made}/{refusal_target}  drops={drops}")

    if kept == 0:
        sys.exit("ABORT: no valid cards produced. Drop reasons: "
                 f"{drops}. Try a stronger teacher, raise --max-new-tokens, or check passages.")
    yield_pct = round(100 * kept / max(1, kept + dropped), 1)
    print(f"==> Wrote {kept} SYNTHETIC rows to {args.out} "
          f"(dropped {dropped}: {drops}; yield {yield_pct}%).")
    print("==> These are machine-generated. Train with: "
          f"scripts/run_train.sh {args.out} synth-run --allow-synthetic")
    print("==> The resulting adapter is a RESEARCH PROTOTYPE — not for clinical use.")


if __name__ == "__main__":
    main()
