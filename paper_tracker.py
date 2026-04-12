"""
paper_tracker.py
----------------
Main entry point. Runs the full pipeline:
  1. Load config (query_config.yaml)
  2. Fetch papers via selected mode:
       Mode A – similarity : Semantic Scholar recommendations API
       Mode B – query      : Semantic Scholar keyword search
  3. Filter against ABS whitelist (only ABS 3+ journals pass)
  4. Summarise + classify each paper by tier (0–4) via LLM
  5. Write a dated JSON data file (YYYY-MM-DD.json)
  6. Call site_generator.py to rebuild index.html + archive.html

No WeChat. No push notifications.
"""

import os
import sys
import json
import re
import datetime
from datetime import timezone
import requests
import yaml

from llm_adapter import get_client, chat
from prompt_refiner import refine_interest_to_prompts, maybe_refresh_prompts, sync_feedback_to_seeds

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

QUERY_CONFIG_FILE  = "query_config.yaml"
LLM_CONFIG_FILE    = "llm_config.yaml"
HISTORY_FILE       = "seen_papers.txt"
BLACKLIST_FILE     = "blacklisted_venues.txt"
WHITELIST_FILE     = "abs_whitelist.txt"
SEED_POSITIVE_FILE = "seed_paper_positive.csv"
SEED_NEGATIVE_FILE = "seed_paper_negative.csv"
MAX_S2_RESULTS     = 100
TOP_N_PAPERS       = 10
ARXIV_TIMEOUT      = 90

S2_BASE = "https://api.semanticscholar.org"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def load_query_config() -> dict:
    if not os.path.exists(QUERY_CONFIG_FILE):
        return {"mode": "query", "keywords": [], "sources": ["semanticscholar"],
                "max_papers": TOP_N_PAPERS}
    with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("mode", "query")
    cfg.setdefault("keywords", [])
    cfg.setdefault("sources", ["semanticscholar"])
    cfg.setdefault("max_papers", TOP_N_PAPERS)
    return cfg


# ---------------------------------------------------------------------------
# Venue normalisation + whitelist filter
# ---------------------------------------------------------------------------

def _normalise_venue(s: str) -> str:
    """Strip punctuation/dots, lowercase, collapse whitespace.

    Lets us match both full names and ISO 4 abbreviations that Semantic
    Scholar returns in the 'venue' field, e.g.:
        "Manag. Sci."           -> "manag sci"
        "Management Science"    -> "management science"
    """
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', s.lower())).strip()


def _filter_and_rank(papers: list, top_n: int) -> list:
    seen_ids    = set(read_lines(HISTORY_FILE))
    blacklisted = [_normalise_venue(v) for v in read_lines(BLACKLIST_FILE)]
    whitelisted = [_normalise_venue(v) for v in read_lines(WHITELIST_FILE)]

    filtered = []
    for p in papers:
        if p.get("paperId") in seen_ids:
            continue
        venue = _normalise_venue(p.get("venue") or "")

        # Whitelist gate: venue must substring-match at least one ABS entry
        # in either direction (handles both full names and ISO 4 abbrev).
        if whitelisted:
            if not any(wv in venue or venue in wv for wv in whitelisted):
                continue

        # Blacklist gate (belt-and-braces)
        if blacklisted and any(bv in venue or venue in bv for bv in blacklisted):
            continue

        if not (p.get("abstract") or "").strip():
            continue
        filtered.append(p)

    return filtered[:top_n]


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _s2_tldr_batch(paper_ids: list) -> dict:
    if not paper_ids:
        return {}
    s2_key  = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}
    url     = f"{S2_BASE}/graph/v1/paper/batch"
    payload = {"ids": paper_ids[:500]}
    params  = {"fields": "paperId,tldr"}
    try:
        r = requests.post(url, headers=headers, params=params,
                          json=payload, timeout=30)
        r.raise_for_status()
        tldr_map = {}
        for item in r.json():
            pid  = item.get("paperId", "")
            tldr = (item.get("tldr") or {}).get("text", "")
            if pid and tldr:
                tldr_map[pid] = tldr
        return tldr_map
    except Exception as exc:
        print(f"  WARNING: TLDR batch fetch failed: {exc}")
        return {}


def _enrich_tldr(papers: list) -> None:
    ids      = [p["paperId"] for p in papers if p.get("paperId")]
    tldr_map = _s2_tldr_batch(ids)
    for p in papers:
        p.setdefault("tldrText", tldr_map.get(p.get("paperId"), ""))


# ---------------------------------------------------------------------------
# Mode A: Semantic Scholar similarity (seed papers)
# ---------------------------------------------------------------------------

def fetch_papers_similarity(cfg: dict) -> list:
    import csv
    s2_key  = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    def read_seed_ids(path):
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    pos_ids = read_seed_ids(SEED_POSITIVE_FILE)
    neg_ids = read_seed_ids(SEED_NEGATIVE_FILE)

    if not pos_ids:
        print("[Similarity] No positive seed papers found. Add IDs to seed_paper_positive.csv")
        return []

    url    = f"{S2_BASE}/recommendations/v1/papers"
    params = {"fields": "paperId,title,abstract,venue,publicationDate,year,authors,externalIds,url",
              "limit": MAX_S2_RESULTS}
    body   = {"positivePaperIds": pos_ids[:100], "negativePaperIds": neg_ids[:100]}

    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=30)
        r.raise_for_status()
        raw = r.json().get("recommendedPapers", [])
        print(f"[Similarity] S2 returned {len(raw)} candidates")
    except Exception as exc:
        print(f"  ERROR: S2 recommendations API failed: {exc}")
        return []

    filtered = _filter_and_rank(raw, cfg.get("max_papers", TOP_N_PAPERS))
    _enrich_tldr(filtered)
    print(f"[Similarity] {len(filtered)} papers passed filter")
    return filtered


# ---------------------------------------------------------------------------
# Mode B: Keyword / query search
# ---------------------------------------------------------------------------

def _s2_keyword_search(query: str, limit: int) -> list:
    """Search S2 filtered to management-relevant fields of study.

    fieldsOfStudy restricts to Business / Economics / Sociology etc.
    so pure ML/NLP benchmark papers don't crowd out management papers.
    publicationTypes=JournalArticle ensures only journal papers (ABS-rankable).
    Computer Science intentionally omitted.
    """
    s2_key  = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    FIELDS_OF_STUDY = ",".join([
        "Business", "Economics", "Sociology",
        "Psychology", "Political Science", "Environmental Science",
    ])

    params = {
        "query":            query,
        "fields":           "paperId,title,abstract,venue,publicationDate,year,authors,externalIds,url",
        "limit":            min(limit, 100),
        "fieldsOfStudy":    FIELDS_OF_STUDY,
        "publicationTypes": "JournalArticle",
    }
    url = f"{S2_BASE}/graph/v1/paper/search"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as exc:
        print(f"  WARNING: S2 search failed for '{query}': {exc}")
        return []


def _arxiv_search(query: str, limit: int) -> list:
    """arXiv search (only used if 'arxiv' is in sources config)."""
    import urllib.parse
    base = "http://export.arxiv.org/api/query"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0, "max_results": limit,
        "sortBy": "submittedDate", "sortOrder": "descending"
    })
    url = f"{base}?{params}"
    try:
        r = requests.get(url, timeout=ARXIV_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:
        print(f"  WARNING: arXiv search failed for '{query}': {exc}")
        return []

    import xml.etree.ElementTree as ET
    ns   = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        pid_url = (entry.findtext("atom:id", "", ns) or "").strip()
        arxiv_id = pid_url.split("/abs/")[-1].replace("/", "_")
        papers.append({
            "paperId":         f"arxiv_{arxiv_id}",
            "title":           (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " "),
            "abstract":        (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " "),
            "venue":           "arXiv",
            "publicationDate": (entry.findtext("atom:published", "", ns) or "")[:10],
            "year":            None,
            "authors":         [{"name": a.findtext("atom:name", "", ns)}
                                for a in entry.findall("atom:author", ns)],
            "externalIds":     {},
            "url":             pid_url,
        })
    return papers


def _update_config_prompts(refined: dict) -> None:
    """Write LLM-refined keywords/prompts back to query_config.yaml."""
    if not os.path.exists(QUERY_CONFIG_FILE):
        return
    try:
        with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["keywords"]      = refined["keywords"]
        cfg["system_prompt"] = refined.get("system_prompt", cfg.get("system_prompt", ""))
        cfg["user_prompt"]   = refined.get("user_prompt",   cfg.get("user_prompt", ""))
        with open(QUERY_CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print("[Query] query_config.yaml updated with refined prompts")
    except (OSError, yaml.YAMLError) as exc:
        print(f"  WARNING: could not persist refined prompts: {exc}")


def _update_config_keywords(keywords: list) -> None:
    """Write ONLY the LLM-refined keywords back to query_config.yaml.
    system_prompt and user_prompt are NOT touched — they are the user's
    authoritative topic definition and must not be overwritten.
    """
    if not os.path.exists(QUERY_CONFIG_FILE):
        return
    try:
        with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["keywords"] = keywords
        with open(QUERY_CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"[Query] query_config.yaml keywords updated: {keywords}")
    except (OSError, yaml.YAMLError) as exc:
        print(f"  WARNING: could not persist refined keywords: {exc}")


def fetch_papers_query(cfg: dict) -> list:
    """Fetch papers using keyword/prompt-based search."""
    client, llm_cfg = get_client()
    interest  = cfg.get("user_interest", "")
    keywords  = cfg.get("keywords", [])
    sources   = cfg.get("sources", ["semanticscholar"])
    max_pap   = cfg.get("max_papers", TOP_N_PAPERS)
    per_query = max(5, MAX_S2_RESULTS // max(len(keywords), 1))

    # If user_interest is set but no keywords yet, refine via LLM
    if interest and not keywords:
        print("[Query] Refining user interest to keywords via LLM...")
        refined  = refine_interest_to_prompts(interest, client=client, cfg=llm_cfg)
        keywords = refined["keywords"]
        # Only keywords are persisted; system/user prompts stay as authored
        _update_config_keywords(keywords)
        print(f"[Query] Refined keywords: {keywords}")

    if not keywords:
        print("  WARNING: No keywords configured. Add keywords to query_config.yaml")
        return []

    raw_papers = []
    seen_in_batch = set()

    for kw in keywords:
        print(f"[Query] Searching: '{kw}'")
        if "semanticscholar" in sources:
            batch = _s2_keyword_search(kw, per_query)
            for p in batch:
                if p.get("paperId") and p["paperId"] not in seen_in_batch:
                    seen_in_batch.add(p["paperId"])
                    raw_papers.append(p)
        if "arxiv" in sources:
            batch = _arxiv_search(kw, per_query)
            for p in batch:
                if p.get("paperId") and p["paperId"] not in seen_in_batch:
                    seen_in_batch.add(p["paperId"])
                    raw_papers.append(p)

    print(f"[Query] Total raw candidates: {len(raw_papers)}")
    filtered = _filter_and_rank(raw_papers, max_pap)
    _enrich_tldr(filtered)
    print(f"[Query] {len(filtered)} papers passed whitelist filter")
    return filtered


# ---------------------------------------------------------------------------
# LLM summarisation + tier classification
# ---------------------------------------------------------------------------

SUMMARISE_SYSTEM = """\
You are a rigorous management science research assistant.
Your task is to classify and summarise academic papers for a daily digest.

CLASSIFICATION TIERS (assign the LOWEST tier the paper qualifies for):

Tier 0 — PRIORITY HIGHLIGHT: paper uses NLP, LLM, corpus linguistics,
  or textual analysis methods AND addresses one or more of these specific
  management science topics: (a) responsibility shifting or accountability
  in organisations, (b) anti-fragility or organisational resilience,
  (c) ESG disclosure or corporate governance, (d) AI adoption in firms.

Tier 1 — Management science paper that uses NLP, corpus linguistics,
  or focuses on computational/textual analysis (but NOT a Tier-0 topic).

Tier 2 — Management science paper using quantitative methods
  (econometrics, surveys, experiments, simulations) — no textual analysis.

Tier 3 — Management science paper that is impactful/highly-cited recently
  but does not fit Tier 0-2.

Tier 4 — Any other management science paper.

OUTPUT: a single valid JSON object — no markdown fences, no preamble — with
exactly these five keys:
  "tier":       integer 0, 1, 2, 3, or 4
  "problem":    one sentence — the management/organisational problem addressed
  "method":     one sentence — the research method used
  "innovation": one sentence — the key contribution
  "topic_tags": list of up to 3 short topic tags relevant to the paper

If the paper is not a management science paper at all, set tier to 4 and
set problem/method/innovation to brief factual descriptions of what it IS."""

SUMMARISE_USER = """\
Classify and summarise the following paper for our management science digest.

Title: {title}
TLDR: {tldr}
Abstract: {abstract}

Apply the tier classification strictly:
- Tier 0: NLP/LLM/textual method + (responsibility shifting OR anti-fragility
  OR ESG OR AI adoption in organisations)
- Tier 1: NLP/LLM/textual method in management science (any topic)
- Tier 2: quantitative method in management science (no textual analysis)
- Tier 3: impactful management science paper (any method)
- Tier 4: any other management science paper

Return JSON with keys: tier, problem, method, innovation, topic_tags"""


def _extract_json(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            cleaned = re.sub(r",\s*([}\]])", r"\1", brace_match.group(0))
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                pass
    return {"problem": raw.strip(), "method": "", "innovation": ""}


def _safe_format(template: str, **kwargs) -> str:
    """Single-pass substitution — all keys replaced simultaneously."""
    if not kwargs:
        return template
    pattern = "|".join(re.escape("{" + k + "}") for k in kwargs)
    return re.sub(pattern, lambda m: str(kwargs[m.group(0)[1:-1]]), template)


def summarise_papers(papers: list, query_cfg: dict) -> list:
    client, llm_cfg = get_client()

    sys_prompt = query_cfg.get("system_prompt") or SUMMARISE_SYSTEM
    user_tmpl  = query_cfg.get("user_prompt")   or SUMMARISE_USER

    results = []
    for idx, p in enumerate(papers):
        title    = p.get("title", "Untitled")
        abstract = (p.get("abstract") or "").strip() or "No abstract available."
        tldr     = (p.get("tldrText") or "").strip() or "N/A"

        doi = ""
        ext = p.get("externalIds")
        if isinstance(ext, dict):
            doi = (ext.get("DOI") or "").strip()
        paper_url = (
            f"https://doi.org/{doi}" if doi
            else p.get("url")
            or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
        )

        authors_list = p.get("authors", [])
        if len(authors_list) > 4:
            names = [authors_list[0].get("name","?"), authors_list[1].get("name","?"),
                     "...",
                     authors_list[-2].get("name","?"), authors_list[-1].get("name","?")]
        else:
            names = [a.get("name","?") for a in authors_list]
        authors_str = ", ".join(names)

        user_msg = _safe_format(user_tmpl, title=title, tldr=tldr, abstract=abstract)

        print(f"[LLM] Summarising [{idx+1}/{len(papers)}]: {title[:70]}...")
        try:
            raw     = chat(sys_prompt, user_msg, client=client, cfg=llm_cfg)
            summary = _extract_json(raw)
        except Exception as exc:
            print(f"  WARNING: LLM call failed for paper {idx+1}: {exc}")
            summary = {"problem": "LLM summarisation failed.", "method": "", "innovation": ""}

        # Extract tier (0–4); default 4 if missing or out-of-range
        try:
            tier = int(summary.get("tier", 4))
            if tier not in (0, 1, 2, 3, 4):
                tier = 4
        except (TypeError, ValueError):
            tier = 4

        topic_tags = summary.get("topic_tags", [])
        if not isinstance(topic_tags, list):
            topic_tags = []

        # Keep summary dict clean — tier and topic_tags live on the paper, not inside summary
        clean_summary = {k: v for k, v in summary.items()
                         if k not in ("tier", "topic_tags")}

        results.append({
            "paperId":    p.get("paperId", ""),
            "title":      title,
            "url":        paper_url,
            "venue":      (p.get("venue") or "Unknown venue").strip(),
            "authors":    authors_str,
            "date":       p.get("publicationDate") or str(p.get("year", "")),
            "abstract":   abstract,
            "tldr":       tldr,
            "summary":    clean_summary,
            "tier":       tier,
            "topic_tags": topic_tags,
        })

    # Sort: tier ascending (0 first), then date descending within each tier
    def _sort_key(p):
        date_str = (p.get("date") or "0000-01-01")[:10].replace("-", "")
        try:
            date_int = int(date_str)
        except ValueError:
            date_int = 0
        return (p["tier"], -date_int)

    results.sort(key=_sort_key)
    return results


# ---------------------------------------------------------------------------
# History update
# ---------------------------------------------------------------------------

def update_history(papers: list):
    if not papers:
        return
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        for p in papers:
            pid = p.get("paperId", "")
            if pid:
                f.write(pid + "\n")
    print(f"[History] Recorded {len(papers)} paper IDs to {HISTORY_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    today = datetime.date.today().isoformat()
    print(f"=== TeamDailyPaper — {today} ===")

    q_cfg = load_query_config()
    mode  = q_cfg["mode"]
    print(f"[Config] Retrieval mode: {mode}")

    if mode == "similarity":
        raw_papers = fetch_papers_similarity(q_cfg)
    elif mode == "query":
        raw_papers = fetch_papers_query(q_cfg)
    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'similarity' or 'query'.")
        sys.exit(1)

    if not raw_papers:
        print("No new papers found today.")
        enriched = []
    else:
        print(f"[Fetch] Retrieved {len(raw_papers)} new papers. Summarising...")
        enriched = summarise_papers(raw_papers, q_cfg)

    data_file = f"{today}.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump({"date": today, "papers": enriched}, f, ensure_ascii=False, indent=2)
    print(f"[Data] Written {data_file}")

    if raw_papers:
        update_history(raw_papers)

    if q_cfg["mode"] == "query":
        print("[Feedback] Checking if prompts need refresh based on user feedback...")
        maybe_refresh_prompts()

    sync_feedback_to_seeds()

    import site_generator
    site_generator.build(today, enriched)
    print("=== Done ===")
