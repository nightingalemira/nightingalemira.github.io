"""
paper_tracker.py
----------------
Main entry point.  Runs the full pipeline:
  1. Load config (query_config.yaml)
  2. Fetch papers via selected mode:
       Mode A – similarity  : Semantic Scholar recommendations API
       Mode B – query       : Semantic Scholar keyword search + optional arXiv
  3. Summarise each paper with the configured LLM (via llm_adapter.py)
  4. Write a dated JSON data file (YYYY-MM-DD.json) to root
  5. Call site_generator.py to rebuild index.html + archive.html

No WeChat, no Server酱, no push notifications.
"""

import os
import sys
import json
import re
import datetime
import requests
import yaml

from llm_adapter import get_client, chat
from prompt_refiner import refine_interest_to_prompts, maybe_refresh_prompts, sync_feedback_to_seeds

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

QUERY_CONFIG_FILE   = "query_config.yaml"
HISTORY_FILE        = "seen_papers.txt"
BLACKLIST_FILE      = "blacklisted_venues.txt"
WHITELIST_FILE      = "abs_whitelist.txt"   # ABS 3+ journal whitelist; empty = allow all
SEED_POSITIVE_FILE  = "seed_paper_positive.csv"
SEED_NEGATIVE_FILE  = "seed_paper_negative.csv"
MAX_S2_RESULTS      = 100
TOP_N_PAPERS        = 10

S2_BASE = "https://api.semanticscholar.org"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


def load_query_config() -> dict:
    cfg = {}
    if os.path.exists(QUERY_CONFIG_FILE):
        with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # Normalise keywords: YAML may deliver a plain string if user forgets list syntax
    raw_keywords = cfg.get("keywords", [])
    if isinstance(raw_keywords, str):
        keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    elif isinstance(raw_keywords, list):
        keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]
    else:
        keywords = []

    return {
        "mode":          cfg.get("mode", "similarity"),          # "similarity" | "query"
        "keywords":      keywords,
        "system_prompt": cfg.get("system_prompt", ""),
        "user_prompt":   cfg.get("user_prompt", ""),
        "user_interest": cfg.get("user_interest", ""),           # freeform interest text
        "max_papers":    int(cfg.get("max_papers", TOP_N_PAPERS)),
        "sources":       cfg.get("sources", ["semanticscholar"]),  # ["semanticscholar","arxiv"]
    }


# ---------------------------------------------------------------------------
# Mode A – Semantic Scholar similarity/recommendations
# ---------------------------------------------------------------------------

def fetch_papers_similarity(cfg: dict) -> list:
    """Use S2 recommendations API with positive/negative seed papers."""
    s2_key = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    positive = read_lines(SEED_POSITIVE_FILE)
    negative = read_lines(SEED_NEGATIVE_FILE)

    print(f"[Similarity] Positive seeds: {len(positive)}, Negative seeds: {len(negative)}")
    if not positive:
        print("ERROR: At least one positive seed paper ID is required for similarity mode.")
        return []

    payload = {
        "positivePaperIds": positive,
        "negativePaperIds": negative,
    }
    params = {
        "fields": "paperId,title,abstract,authors,url,venue,externalIds,publicationDate,year",
        "limit":  MAX_S2_RESULTS,
    }
    url = f"{S2_BASE}/recommendations/v1/papers"
    resp = requests.post(url, json=payload, headers=headers, params=params, timeout=30)

    if resp.status_code != 200:
        print(f"ERROR: S2 recommendations API returned {resp.status_code}: {resp.text[:200]}")
        return []

    raw = resp.json().get("recommendedPapers", [])
    return _filter_and_rank(raw, cfg["max_papers"])


# ---------------------------------------------------------------------------
# Mode B – Direct keyword/prompt query
# ---------------------------------------------------------------------------

def fetch_papers_query(cfg: dict) -> list:
    """
    Search S2 and/or arXiv using directives derived from the user's freeform
    research interest.  The pipeline is:

      1. If user_interest is set in query_config.yaml, ask the LLM to convert
         it into precise keywords + focused summarisation prompts, then cache
         them back into the config so subsequent runs can run without an LLM
         call until the interest or feedback changes.
      2. Fall back to any explicit keywords already in the config.
      3. Fall back to user_prompt text as the raw query string.
      4. Search S2 and/or arXiv with the resolved keywords.
    """
    client, llm_cfg = get_client()

    interest   = cfg.get("user_interest", "").strip()
    keywords   = cfg.get("keywords", [])
    user_prompt_raw = cfg.get("user_prompt", "")

    # ── Step 1: LLM refinement from freeform interest ─────────────────────
    if interest:
        print(f"[Query] Refining freeform interest via LLM: {interest[:80]}...")
        refined = refine_interest_to_prompts(interest, client=client, cfg=llm_cfg)
        keywords = refined["keywords"]
        # Write ONLY the refined keywords back to config (prompts are fixed by the user
        # in query_config.yaml and must not be overwritten by the LLM refiner).
        _update_config_keywords(keywords)
        # Do NOT touch cfg["system_prompt"] or cfg["user_prompt"] here — they are
        # read from query_config.yaml and stay as the user set them.
        print(f"[Query] Refined keywords: {keywords}")

    # ── Step 2 / 3: fall back to existing keywords or raw user_prompt ─────
    if not keywords:
        keywords = [user_prompt_raw] if user_prompt_raw else []
    if not keywords:
        print("ERROR: query mode requires user_interest, keywords, or user_prompt in query_config.yaml")
        return []

    query_str = " ".join(keywords)
    sources   = cfg.get("sources", ["semanticscholar"])
    papers    = []

    if "semanticscholar" in sources:
        papers += _s2_keyword_search(query_str, cfg["max_papers"] * 3)
    if "arxiv" in sources:
        papers += _arxiv_search(query_str, cfg["max_papers"] * 2)

    # Deduplicate by title (lower-cased)
    seen_titles = set()
    deduped = []
    for p in papers:
        t = (p.get("title") or "").lower().strip()
        if t and t not in seen_titles:
            seen_titles.add(t)
            deduped.append(p)

    return _filter_and_rank(deduped, cfg["max_papers"])


def _update_config_prompts(refined: dict) -> None:
    """Write LLM-refined keywords/prompts back to query_config.yaml."""
    if not os.path.exists(QUERY_CONFIG_FILE):
        return
    try:
        with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["keywords"]      = refined["keywords"]
        cfg["system_prompt"] = refined.get("system_prompt", cfg.get("system_prompt",""))
        cfg["user_prompt"]   = refined.get("user_prompt",   cfg.get("user_prompt",""))
        with open(QUERY_CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"[Query] query_config.yaml updated with refined prompts")
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


def _s2_keyword_search(query: str, limit: int) -> list:
    """Search S2 filtered to management-relevant fields of study.

    fieldsOfStudy restricts results to Business / Economics / Sociology etc.
    so that pure ML/NLP benchmark papers don't crowd out management papers.
    Cross-disciplinary NLP+management papers are indexed under Business or
    Economics in S2, so they are still returned.
    publicationTypes=JournalArticle ensures only journal papers come back —
    which aligns with the ABS whitelist (conference papers are not ABS-ranked).
    """
    s2_key = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    # S2 field-of-study slugs covering management / business / social science.
    # "Computer Science" is intentionally omitted — pure CS/ML papers should not
    # dominate. Cross-disciplinary work appears under Business or Economics.
    FIELDS_OF_STUDY = ",".join([
        "Business",
        "Economics",
        "Sociology",
        "Psychology",
        "Political Science",
        "Environmental Science",
    ])

    params = {
        "query":            query,
        "fields":           ("paperId,title,abstract,authors,url,venue,"
                             "externalIds,publicationDate,year"),
        "fieldsOfStudy":    FIELDS_OF_STUDY,
        "publicationTypes": "JournalArticle",
        "limit":            min(limit, 100),
    }
    url = f"{S2_BASE}/graph/v1/paper/search"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"WARNING: S2 search returned {resp.status_code}: {resp.text[:200]}")
        return []
    return resp.json().get("data", [])


def _arxiv_search(query: str, max_results: int) -> list:
    """Fetch from arXiv Atom feed and normalise to S2-like dicts.

    Retries up to 3 times with exponential back-off (10 s, 20 s, 40 s).
    On persistent failure returns [] so the rest of the pipeline still runs.
    """
    import xml.etree.ElementTree as ET
    import time
    from requests.exceptions import ReadTimeout, ConnectionError as ReqConnError

    encoded = requests.utils.quote(query)
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    ARXIV_TIMEOUT = 90          # seconds — arXiv is slow under load on CI runners
    MAX_RETRIES   = 3
    BACKOFF_BASE  = 10          # seconds; doubles each retry: 10, 20, 40

    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=ARXIV_TIMEOUT)
            break                   # success — exit retry loop
        except (ReadTimeout, ReqConnError) as exc:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"WARNING: arXiv attempt {attempt} failed ({exc.__class__.__name__}). "
                      f"Retrying in {wait} s…")
                time.sleep(wait)
            else:
                print(f"WARNING: arXiv search failed after {MAX_RETRIES} attempts "
                      f"({exc.__class__.__name__}). Skipping arXiv results for today.")
                return []

    if resp.status_code != 200:
        print(f"WARNING: arXiv search returned {resp.status_code}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        print(f"WARNING: arXiv returned malformed XML: {exc}")
        return []
    papers = []
    for entry in root.findall("atom:entry", ns):
        title    = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
        pub_date = (entry.findtext("atom:published", "", ns) or "")[:10]
        arxiv_id = re.sub(
            r"v\d+$", "",
            (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
        )
        link     = f"https://arxiv.org/abs/{arxiv_id}"
        authors  = [
            {"name": a.findtext("atom:name", "", ns)}
            for a in entry.findall("atom:author", ns)
        ]
        papers.append({
            "paperId":         f"arxiv:{arxiv_id}",
            "title":           title,
            "abstract":        abstract,
            "authors":         authors,
            "url":             link,
            "venue":           "arXiv",
            "externalIds":     {"ArXiv": arxiv_id},
            "publicationDate": pub_date,
            "year":            int(pub_date[:4]) if pub_date else None,
            "tldrText":        "",
        })
    return papers


# ---------------------------------------------------------------------------
# Shared filter / rank
# ---------------------------------------------------------------------------

def _normalise_venue(s: str) -> str:
    """Strip punctuation/dots, lowercase, collapse whitespace.

    This normalisation lets us match both full names and ISO 4 abbreviations
    that Semantic Scholar returns in the 'venue' field, e.g.:
        "Manag. Sci."  →  "manag sci"
        "Management Science"  →  "management science"
    A whitelist entry "manag sci" is then a substring of "management science"
    (or vice-versa), so either form matches.
    """
    import re as _re
    return _re.sub(r'\s+', ' ', _re.sub(r'[^a-z0-9\s]', '', s.lower())).strip()


def _filter_and_rank(papers: list, top_n: int) -> list:
    seen_ids    = set(read_lines(HISTORY_FILE))
    blacklisted = [_normalise_venue(v) for v in read_lines(BLACKLIST_FILE)]
    # Build normalised whitelist: both full names and ISO 4 abbreviations are in
    # abs_whitelist.txt, so a single normalised substring check in either
    # direction covers exact names, abbreviated names, and edge cases.
    whitelisted = [_normalise_venue(v) for v in read_lines(WHITELIST_FILE)]

    filtered = []
    for p in papers:
        if p.get("paperId") in seen_ids:
            continue
        venue = _normalise_venue(p.get("venue") or "")

        # Whitelist gate: if whitelist is non-empty, venue must substring-match
        # at least one entry in either direction.  Non-journal venues (arXiv,
        # NeurIPS, workshop proceedings, etc.) will not match any ABS entry
        # and are therefore blocked automatically.
        if whitelisted:
            if not any(wv in venue or venue in wv for wv in whitelisted):
                continue

        # Blacklist gate (secondary; belt-and-braces for edge cases)
        if blacklisted and any(bv in venue or venue in bv for bv in blacklisted):
            continue

        if not (p.get("abstract") or "").strip():
            continue
        filtered.append(p)

    # Sort newest-first
    def _date_key(p):
        d = p.get("publicationDate") or ""
        if d:
            return d
        y = p.get("year")
        return f"{y}-12-31" if y else "1900-01-01"

    filtered.sort(key=_date_key, reverse=True)
    top = filtered[:top_n]

    # Enrich with TLDR via S2 batch (best-effort)
    _enrich_tldr(top)
    return top


def _enrich_tldr(papers: list):
    """Batch-fetch TLDR from S2 for papers that have a real S2 paperId."""
    s2_key  = os.getenv("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    s2_papers = [p for p in papers if not p.get("paperId", "").startswith("arxiv:")]
    if not s2_papers:
        for p in papers:
            p.setdefault("tldrText", "")
        return

    ids = [p["paperId"] for p in s2_papers]
    resp = requests.post(
        f"{S2_BASE}/graph/v1/paper/batch",
        json={"ids": ids},
        headers=headers,
        params={"fields": "paperId,tldr"},
        timeout=30,
    )
    tldr_map = {}
    if resp.status_code == 200:
        for item in resp.json():
            if item and isinstance(item.get("tldr"), dict):
                tldr_map[item["paperId"]] = (item["tldr"].get("text") or "").strip()

    for p in papers:
        p.setdefault("tldrText", tldr_map.get(p.get("paperId"), ""))


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------

SUMMARISE_SYSTEM = """\
You are a rigorous academic expert. Given a paper title, TLDR, and abstract, \
produce a concise structured summary in English. \
Output ONLY a JSON object with these exact keys:
  problem   – one sentence: the pain point or research gap addressed
  method    – the core architecture, algorithm, model, or mechanism used
  innovation – the key improvement or contribution (metrics, limits resolved)
No extra keys, no markdown fences, no preamble."""

SUMMARISE_USER = """\
Title: {title}
TLDR: {tldr}
Abstract: {abstract}"""



def _extract_json(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output.
    Handles: code fences, preamble text, trailing commas, single backticks."""
    # 1. Try direct parse first (clean output, no fences)
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Extract from ```json ... ``` or ``` ... ``` fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # 3. Grab the first {...} block anywhere (handles preamble text)
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            # 4. Try removing trailing commas (common LLM mistake)
            cleaned = re.sub(r",\s*([}\]])", r"\1", brace_match.group(0))
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                pass
    # 5. Give up — store the raw text as the problem field
    return {"problem": raw.strip(), "method": "", "innovation": ""}


def _safe_format(template: str, **kwargs) -> str:
    """Single-pass substitution — all keys are replaced simultaneously so
    placeholder text inside values is never re-interpreted as a template token."""
    if not kwargs:
        return template
    pattern = "|".join(re.escape("{" + k + "}") for k in kwargs)
    return re.sub(pattern, lambda m: str(kwargs[m.group(0)[1:-1]]), template)

def summarise_papers(papers: list, query_cfg: dict) -> list:
    client, llm_cfg = get_client()

    # Allow query_config.yaml to override the system/user prompts
    sys_prompt  = query_cfg.get("system_prompt") or SUMMARISE_SYSTEM
    user_tmpl   = query_cfg.get("user_prompt")   or SUMMARISE_USER

    results = []
    for idx, p in enumerate(papers):
        title    = p.get("title", "Untitled")
        abstract = (p.get("abstract") or "").strip() or "No abstract available."
        tldr     = (p.get("tldrText") or "").strip() or "N/A"

        # Build DOI / URL
        doi = ""
        ext = p.get("externalIds")
        if isinstance(ext, dict):
            doi = (ext.get("DOI") or "").strip()
        paper_url = (
            f"https://doi.org/{doi}" if doi
            else p.get("url")
            or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
        )

        # Authors (truncate if > 4)
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
            raw = chat(sys_prompt, user_msg, client=client, cfg=llm_cfg)
            summary = _extract_json(raw)
        except Exception as exc:
            print(f"  WARNING: LLM call failed for paper {idx+1} ({title[:50]}): {exc}")
            summary = {"problem": "LLM summarisation failed.", "method": "", "innovation": ""}

        results.append({
            "paperId":   p.get("paperId", ""),
            "title":     title,
            "url":       paper_url,
            "venue":     (p.get("venue") or "Unknown venue").strip(),
            "authors":   authors_str,
            "date":      p.get("publicationDate") or str(p.get("year", "")),
            "abstract":  abstract,
            "tldr":      tldr,
            "summary":   summary,
        })

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
        # Still regenerate site (shows empty state)
        enriched = []
    else:
        print(f"[Fetch] Retrieved {len(raw_papers)} new papers. Summarising...")
        enriched = summarise_papers(raw_papers, q_cfg)

    # Write dated JSON data file (before updating history, so a crash here is safe)
    data_file = f"{today}.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump({"date": today, "papers": enriched}, f, ensure_ascii=False, indent=2)
    print(f"[Data] Written {data_file}")

    if raw_papers:
        update_history(raw_papers)

    # Refresh prompts based on accumulated feedback (query mode)
    if q_cfg["mode"] == "query":
        print("[Feedback] Checking if prompts need refresh based on user feedback...")
        maybe_refresh_prompts()

    # Sync liked/disliked papers to similarity seeds (both modes)
    sync_feedback_to_seeds()

    # Regenerate website
    import site_generator
    site_generator.build(today, enriched)
    print("=== Done ===")
