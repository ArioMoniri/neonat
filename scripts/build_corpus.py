#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_corpus.py — pull OPEN neonatology/perinatology passages for distillation.
================================================================================
Builds data/corpus/passages.jsonl, one passage per line:
    {"passage_id","source","license","url","topic","passage"}

Sources (each is independent + best-effort; a failing source is skipped, not fatal):
  • europepmc — Europe PMC OPEN-ACCESS full text (license-tracked)
  • pubmed    — PubMed abstracts via NCBI E-utilities
  • urls      — WHO / public guideline pages or PDFs you list (--urls file or --url)
  • exa       — optional web search (needs EXA_API_KEY)

Only the Europe PMC OPEN-ACCESS subset is fetched in full text; abstracts are
public. Per-passage source/license/url are recorded so downstream use is auditable.

Usage:
    python build_corpus.py --out data/corpus/passages.jsonl \
        --sources europepmc,pubmed,urls --limit 400 \
        --urls data/corpus/guideline_urls.txt
    python build_corpus.py --selftest --out data/corpus/passages.jsonl   # offline stub
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "neoperi-cdss-corpus/1.0 (research; contact: local)"}

DEFAULT_TOPICS = [
    "neonatal hyperbilirubinemia management",
    "neonatal sepsis evaluation newborn",
    "early onset neonatal sepsis risk",
    "late onset neonatal sepsis",
    "preterm birth management guideline",
    "antenatal corticosteroids preterm",
    "neonatal resuscitation",
    "delayed cord clamping newborn",
    "respiratory distress syndrome newborn surfactant",
    "bronchopulmonary dysplasia preterm",
    "neonatal hypoglycemia",
    "perinatal asphyxia hypoxic ischemic encephalopathy therapeutic hypothermia",
    "necrotizing enterocolitis preterm",
    "patent ductus arteriosus preterm",
    "intraventricular hemorrhage preterm",
    "retinopathy of prematurity screening",
    "maternal group B streptococcus newborn",
    "neonatal jaundice phototherapy exchange transfusion",
    "neonatal abstinence syndrome",
    "congenital heart disease newborn screening",
    "neonatal encephalopathy seizures",
    "preterm nutrition enteral feeding",
    "neonatal hypothermia thermoregulation",
    "maternal preeclampsia fetal",
    "gestational diabetes neonatal outcomes",
    "chorioamnionitis neonatal",
    "neonatal anemia transfusion",
    "neonatal thrombocytopenia",
    "congenital infection TORCH newborn",
    "apnea of prematurity caffeine",
]
# A passage must look neonatal/perinatal to be kept.
RELEVANCE = re.compile(
    r"neonat|newborn|infant|preterm|premature|gestational|perinat|"
    r"nicu|fetal|fetus|maternal|yenidoğan|prematür|gebelik|perinat",
    re.IGNORECASE)


# ----------------------------------------------------------------------------
def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _chunks(text, target_words=180, max_words=320):
    """Split prose into ~target_words passages on sentence-ish boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= target_words:
            chunk = " ".join(cur).strip()
            if len(chunk.split()) <= max_words:
                out.append(chunk)
            cur = []
    if cur:
        chunk = " ".join(cur).strip()
        if len(chunk.split()) >= 40:
            out.append(chunk)
    return out


def _localname(el):
    return el.tag.split("}")[-1].lower()


def _jats_extract(root):
    """Return (body+abstract paragraph text, license_string) from JATS XML,
    scoped to <body>/<abstract> and EXCLUDING <back>/<ref-list> boilerplate."""
    # License from <permissions>/<license>.
    license_ = "unknown"
    for el in root.iter():
        if _localname(el) == "license":
            lt = el.get("license-type") or ""
            href = ""
            for a in el.iter():
                href = a.get("{http://www.w3.org/1999/xlink}href") or href
            txt = " ".join(el.itertext()).strip()
            # Prefer the actual license URL (e.g. a CC-BY link) over the generic
            # "open-access" license-type attribute.
            license_ = (href or lt or txt[:80] or "open-access").strip()
            break
    # Body + abstract paragraphs only.
    paras = []
    for el in root.iter():
        if _localname(el) in ("body", "abstract"):
            for p in el.iter():
                if _localname(p) == "p":
                    t = " ".join(p.itertext()).strip()
                    if t:
                        paras.append(t)
    return "\n".join(paras), license_


def _emit(seen, rows, source, license_, url, topic, text):
    import hashlib
    for ch in _chunks(text):
        if not RELEVANCE.search(ch):
            continue
        key = hashlib.sha256(ch.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "passage_id": f"{source}-{len(rows) + 1:05d}",
            "source": source, "license": license_, "url": url,
            "topic": topic, "passage": ch,
        })


# ----------------------------------------------------------------------------
def src_europepmc(rows, seen, topics, per_topic, pause, max_total):
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    for topic in topics:
        if len(rows) >= max_total:
            return
        try:
            q = urllib.parse.quote(f'{topic} AND OPEN_ACCESS:y AND IN_EPMC:y')
            url = f"{base}/search?query={q}&format=json&pageSize={per_topic}&resultType=lite"
            hits = json.loads(_get(url)).get("resultList", {}).get("result", [])
        except Exception as e:  # noqa: BLE001
            print(f"    [europepmc] search failed for '{topic}': {e}")
            continue
        for h in hits:
            if len(rows) >= max_total:
                return
            pmcid = h.get("pmcid")
            if not pmcid or h.get("isOpenAccess") != "Y":
                continue
            art_url = f"https://europepmc.org/article/PMC/{pmcid}"
            try:
                root = ET.fromstring(_get(f"{base}/{pmcid}/fullTextXML"))
                text, lic = _jats_extract(root)          # real license + body-scoped
            except Exception as e:  # noqa: BLE001
                print(f"    [europepmc] fulltext {pmcid} failed: {e}")
                continue
            if not text.strip():
                continue
            _emit(seen, rows, "europepmc", lic or h.get("license", "open-access"),
                  art_url, topic, text)
            time.sleep(pause)
        print(f"    [europepmc] '{topic}': corpus now {len(rows)} passages")
        if len(rows) >= max_total:
            return


def src_pubmed(rows, seen, topics, per_topic, pause, max_total):
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    for topic in topics:
        if len(rows) >= max_total:
            return
        try:
            q = urllib.parse.quote(topic)
            s = json.loads(_get(
                f"{eutils}/esearch.fcgi?db=pubmed&retmode=json&retmax={per_topic}&term={q}"))
            ids = s.get("esearchresult", {}).get("idlist", [])
        except Exception as e:  # noqa: BLE001
            print(f"    [pubmed] esearch failed for '{topic}': {e}")
            continue
        if not ids:
            continue
        try:
            xml = _get(f"{eutils}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(ids)}")
            root = ET.fromstring(xml)
        except Exception as e:  # noqa: BLE001
            print(f"    [pubmed] efetch failed for '{topic}': {e}")
            continue
        for art in root.iter("PubmedArticle"):
            if len(rows) >= max_total:
                return
            pmid = art.findtext(".//PMID") or "?"
            # itertext() keeps text inside <i>/<sup> etc. and structured-abstract labels.
            abstract = " ".join(" ".join(t.itertext()) for t in art.iter("AbstractText")).strip()
            if not abstract:
                continue
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            _emit(seen, rows, "pubmed", "abstract (PubMed)", url, topic, abstract)
        print(f"    [pubmed] '{topic}': corpus now {len(rows)} passages")
        time.sleep(pause)


def src_urls(rows, seen, urls, pause, max_total):
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001
        BeautifulSoup = None
    for url in urls:
        if len(rows) >= max_total:
            return
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        try:
            raw = _get(url, timeout=60)
        except Exception as e:  # noqa: BLE001
            print(f"    [urls] fetch failed {url}: {e}")
            continue
        if url.lower().endswith(".pdf") or raw[:5] == b"%PDF-":
            try:
                from pypdf import PdfReader
                import io
                text = "\n".join((pg.extract_text() or "")
                                 for pg in PdfReader(io.BytesIO(raw)).pages)
            except Exception as e:  # noqa: BLE001
                print(f"    [urls] pdf parse failed {url}: {e}")
                continue
        else:
            html = raw.decode("utf-8", "ignore")
            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, "html.parser")
                for t in soup(["script", "style", "nav", "header", "footer"]):
                    t.extract()
                text = soup.get_text(" ")
            else:
                text = re.sub(r"<[^>]+>", " ", html)
        _emit(seen, rows, "url", "user-provided", url, "user", text)
        print(f"    [urls] {url}: corpus now {len(rows)} passages")
        time.sleep(pause)


def src_exa(rows, seen, topics, per_topic, pause, max_total):
    key = os.environ.get("EXA_API_KEY")
    if not key:
        print("    [exa] EXA_API_KEY not set — skipping.")
        return
    for topic in topics:
        if len(rows) >= max_total:
            return
        try:
            body = json.dumps({"query": topic, "numResults": per_topic,
                               "contents": {"text": True}}).encode()
            req = urllib.request.Request(
                "https://api.exa.ai/search", data=body,
                headers={**UA, "x-api-key": key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                results = json.loads(r.read()).get("results", [])
        except Exception as e:  # noqa: BLE001
            print(f"    [exa] search failed for '{topic}': {e}")
            continue
        for res in results:
            text = res.get("text") or ""
            url = res.get("url", "")
            if text:
                _emit(seen, rows, "exa", "web (see url)", url, topic, text)
        print(f"    [exa] '{topic}': corpus now {len(rows)} passages")
        time.sleep(pause)


# ----------------------------------------------------------------------------
def selftest(out):
    rows = [{
        "passage_id": "selftest-00001", "source": "selftest", "license": "synthetic",
        "url": "", "topic": "neonatal jaundice",
        "passage": ("Neonatal jaundice is common in newborns. Total serum bilirubin "
                    "should be interpreted against the infant's age in hours and "
                    "gestational age. Risk factors include prematurity, hemolysis, and "
                    "exclusive breastfeeding with inadequate intake. Phototherapy "
                    "thresholds depend on gestational age and risk factors."),
    }]
    _write(out, rows)
    print(f"==> [selftest] wrote {len(rows)} stub passage(s) to {out}")


def _write(out, rows):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build an open neoperi passage corpus.")
    ap.add_argument("--out", default="data/corpus/passages.jsonl")
    ap.add_argument("--sources", default="europepmc,pubmed",
                    help="comma list of: europepmc,pubmed,urls,exa")
    ap.add_argument("--limit", type=int, default=400, help="max total passages")
    ap.add_argument("--per-topic", type=int, default=8, help="articles fetched per topic")
    ap.add_argument("--pause", type=float, default=0.34, help="seconds between API calls")
    ap.add_argument("--urls", default=None, help="file with one URL per line (for 'urls')")
    ap.add_argument("--url", action="append", default=[], help="a URL (repeatable)")
    ap.add_argument("--topics", default=None, help="file with one topic/query per line")
    ap.add_argument("--selftest", action="store_true", help="offline: write a stub corpus")
    ap.add_argument("--accept-unvetted-license", action="store_true",
                    help="required to include 'urls'/'exa' sources, whose text is NOT "
                         "license-vetted — you confirm you have the right to use it")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.out)
        return

    topics = DEFAULT_TOPICS
    if args.topics and os.path.exists(args.topics):
        topics = [l.strip() for l in open(args.topics, encoding="utf-8") if l.strip()]

    urls = list(args.url)
    if args.urls and os.path.exists(args.urls):
        urls += [l.strip() for l in open(args.urls, encoding="utf-8") if l.strip()]

    rows, seen = [], set()
    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
    unvetted = [s for s in wanted if s in ("urls", "exa")]
    if unvetted and not args.accept_unvetted_license:
        print(f"    [skip] {unvetted} carry NO license vetting; re-run with "
              "--accept-unvetted-license to include them (you confirm usage rights).")
        wanted = [s for s in wanted if s not in ("urls", "exa")]
    print(f"==> Building corpus from sources={wanted} (limit {args.limit})")
    if "europepmc" in wanted:
        src_europepmc(rows, seen, topics, args.per_topic, args.pause, args.limit)
    if "pubmed" in wanted:
        src_pubmed(rows, seen, topics, args.per_topic, args.pause, args.limit)
    if "urls" in wanted:
        if not urls:
            print("    [urls] no URLs provided (--urls/--url) — skipping.")
        else:
            src_urls(rows, seen, urls, args.pause, args.limit)
    if "exa" in wanted:
        src_exa(rows, seen, topics, args.per_topic, args.pause, args.limit)

    if not rows:
        sys.exit("ABORT: no passages collected. Check network/sources, or use --selftest.")
    rows = rows[:args.limit]
    _write(args.out, rows)
    by_src = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"==> Wrote {len(rows)} passages to {args.out}  by source: {by_src}")


if __name__ == "__main__":
    main()
