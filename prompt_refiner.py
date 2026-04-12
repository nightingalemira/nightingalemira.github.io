"""
prompt_refiner.py
-----------------
Converts any freeform user input (fragments, sentences, research descriptions)
into precise API search terms and LLM prompts.  Also reads the daily feedback
file (feedback/YYYY-MM-DD.json) to learn thumbs-up / thumbs-down preferences
and updates query_config.yaml when preferences change.

Called by paper_tracker.py during the query pipeline.
"""

import os
import json
import re
import glob
import datetime
import yaml

from llm_adapter import get_client, chat

FEEDBACK_DIR      = "feedback"
QUERY_CONFIG_FILE = "query_config.yaml"
PREFERENCE_FILE   = "preference_state.json"   # persists accumulated signal


# ---------------------------------------------------------------------------
# Step 1: Turn any freeform user text → search keywords + LLM prompts
# ---------------------------------------------------------------------------

REFINE_SYSTEM = """\
You are a research librarian assistant. The user describes their research interest
in any form — a sentence, a topic fragment, bullet notes, or even rough ideas.
Your job is to distil this into actionable search directives.

Output ONLY a JSON object with these exact keys:
  keywords   – list of 3-7 short search phrases for the paper APIs (each ≤5 words)
  system_prompt – one paragraph instructing the LLM summariser what to focus on
  user_prompt   – summarisation template using {title}, {tldr}, {abstract} that
                  extracts information most relevant to the user's interest
No extra keys, no markdown fences, no preamble."""

REFINE_USER = "User research interest:\n{interest}"


def refine_interest_to_prompts(interest: str, client=None, cfg=None) -> dict:
    """
    Takes any freeform description of the user's research interest and returns
    a dict with keys: keywords (list), system_prompt (str), user_prompt (str).
    Falls back to simple keyword extraction if LLM call fails.
    """
    if client is None or cfg is None:
        client, cfg = get_client()

    from paper_tracker import _safe_format, _extract_json

    user_msg = _safe_format(REFINE_USER, interest=interest)
    try:
        raw = chat(REFINE_SYSTEM, user_msg, client=client, cfg=cfg)
        result = _extract_json(raw)
        # Validate structure
        if not isinstance(result.get("keywords"), list) or not result["keywords"]:
            raise ValueError("keywords missing or empty")
        return {
            "keywords":      [str(k).strip() for k in result["keywords"] if str(k).strip()],
            "system_prompt": str(result.get("system_prompt") or "").strip(),
            "user_prompt":   str(result.get("user_prompt") or "").strip(),
        }
    except Exception as exc:
        print(f"  [Refiner] LLM refinement failed ({exc}); falling back to naive extraction")
        return _naive_keyword_extract(interest)


def _naive_keyword_extract(text: str) -> dict:
    """Fallback: split on punctuation, take longest meaningful phrases."""
    # Remove common stop words and punctuation
    stop = {"a","an","the","and","or","but","in","on","at","to","for","of","with",
            "is","are","was","were","be","been","i","my","our","we","that","this",
            "it","its","about","some","any","can","could","should","would","have"}
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    kept = [w for w in words if w not in stop]
    # Build bigrams and trigrams
    phrases = []
    for n in (3, 2, 1):
        for i in range(len(kept) - n + 1):
            phrases.append(" ".join(kept[i:i+n]))
    # Deduplicate while preserving order
    seen, unique = set(), []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    keywords = unique[:6] if unique else [text[:50]]
    from paper_tracker import SUMMARISE_SYSTEM, SUMMARISE_USER
    return {
        "keywords":      keywords,
        "system_prompt": SUMMARISE_SYSTEM,
        "user_prompt":   SUMMARISE_USER,
    }


# ---------------------------------------------------------------------------
# Step 2: Load accumulated feedback from all past feedback files
# ---------------------------------------------------------------------------

def load_all_feedback() -> dict:
    """
    Reads all feedback/YYYY-MM-DD.json files and returns aggregated signals:
      liked_titles    – list of paper titles the user thumbed up
      disliked_titles – list of paper titles the user thumbed down
      liked_ids       – set of paper IDs (S2 or arxiv) that were liked
      disliked_ids    – set of paper IDs that were disliked
    """
    liked_titles, disliked_titles = [], []
    liked_ids, disliked_ids = set(), set()

    if not os.path.exists(FEEDBACK_DIR):
        return {"liked_titles": [], "disliked_titles": [],
                "liked_ids": set(), "disliked_ids": set()}

    for fpath in sorted(glob.glob(os.path.join(FEEDBACK_DIR, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("feedback", []):
                pid   = entry.get("paperId", "")
                title = entry.get("title", "")
                vote  = entry.get("vote", "")   # "up" | "down"
                if vote == "up":
                    liked_ids.add(pid)
                    if title: liked_titles.append(title)
                elif vote == "down":
                    disliked_ids.add(pid)
                    if title: disliked_titles.append(title)
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            continue

    return {
        "liked_titles":    liked_titles[-30:],   # keep most recent 30
        "disliked_titles": disliked_titles[-30:],
        "liked_ids":       liked_ids,
        "disliked_ids":    disliked_ids,
    }


# ---------------------------------------------------------------------------
# Step 3: Decide whether prompts need refreshing and update query_config.yaml
# ---------------------------------------------------------------------------

REFRESH_SYSTEM = """\
You are a research librarian assistant helping refine a daily paper recommendation system.
Below you will see:
  - The user's stated research interest (original freeform text)
  - Papers the user thumbed UP (relevant, interesting)
  - Papers the user thumbed DOWN (irrelevant, not useful)

Based on the feedback patterns, produce updated search directives that will
surface more papers like the liked ones and fewer like the disliked ones.

Output ONLY a JSON object with these exact keys:
  keywords      – updated list of 3-7 short search phrases (each ≤5 words)
  system_prompt – updated summariser instruction paragraph
  user_prompt   – updated summarisation template using {title}, {tldr}, {abstract}
  changed       – boolean: true if the directives meaningfully changed vs. current
No extra keys, no markdown fences, no preamble."""


def maybe_refresh_prompts(client=None, cfg_llm=None) -> bool:
    """
    Reads the current query_config.yaml, loads all feedback, asks the LLM
    whether the prompts/keywords should change, and if so, writes updated
    values back to query_config.yaml.

    Returns True if query_config.yaml was updated, False otherwise.
    """
    if client is None or cfg_llm is None:
        client, cfg_llm = get_client()

    # Load current config
    if not os.path.exists(QUERY_CONFIG_FILE):
        print("  [Refiner] query_config.yaml not found; skipping refresh")
        return False

    with open(QUERY_CONFIG_FILE, "r", encoding="utf-8") as f:
        raw_yaml = f.read()
        cfg = yaml.safe_load(raw_yaml) or {}

    if cfg.get("mode") != "query":
        return False   # only applies to query mode

    current_interest = cfg.get("user_interest", "")
    current_keywords = cfg.get("keywords", [])
    if not current_interest:
        print("  [Refiner] No user_interest set in query_config.yaml; skipping refresh")
        return False

    fb = load_all_feedback()
    if not fb["liked_titles"] and not fb["disliked_titles"]:
        print("  [Refiner] No feedback collected yet; skipping refresh")
        return False

    liked_str    = "\n".join(f"  + {t}" for t in fb["liked_titles"][-15:]) or "  (none)"
    disliked_str = "\n".join(f"  - {t}" for t in fb["disliked_titles"][-15:]) or "  (none)"
    kw_str       = ", ".join(current_keywords)

    from paper_tracker import _extract_json
    refresh_user = (
        f"User's stated interest:\n  {current_interest}\n\n"
        f"Current search keywords: {kw_str}\n\n"
        f"Papers user LIKED (thumbs up):\n{liked_str}\n\n"
        f"Papers user DISLIKED (thumbs down):\n{disliked_str}"
    )

    try:
        raw = chat(REFRESH_SYSTEM, refresh_user, client=client, cfg=cfg_llm)
        result = _extract_json(raw)
    except Exception as exc:
        print(f"  [Refiner] Refresh LLM call failed: {exc}")
        return False

    if not result.get("changed", False):
        print("  [Refiner] LLM determined no meaningful change needed; keeping current prompts")
        return False

    new_kw  = result.get("keywords", current_keywords)
    new_sys = result.get("system_prompt", cfg.get("system_prompt", ""))
    new_usr = result.get("user_prompt",   cfg.get("user_prompt", ""))

    if not isinstance(new_kw, list) or not new_kw:
        print("  [Refiner] Refresh returned invalid keywords; aborting update")
        return False

    # Patch query_config.yaml in place, preserving all comments
    cfg["keywords"]      = [str(k).strip() for k in new_kw if str(k).strip()]
    cfg["system_prompt"] = new_sys
    cfg["user_prompt"]   = new_usr

    # Re-serialise: preserve comment header, replace the YAML body
    with open(QUERY_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"  [Refiner] query_config.yaml updated with new keywords: {cfg['keywords']}")
    return True


# ---------------------------------------------------------------------------
# Step 4: For similarity mode — promote liked papers, demote disliked ones
# ---------------------------------------------------------------------------

def sync_feedback_to_seeds() -> None:
    """
    For similarity mode: appends liked paper IDs to seed_paper_positive.csv
    and disliked IDs to seed_paper_negative.csv (if not already present).
    Called at end of daily run regardless of mode.
    """
    fb = load_all_feedback()
    if not fb["liked_ids"] and not fb["disliked_ids"]:
        return

    def _add_ids(filepath: str, new_ids: set) -> int:
        existing = set()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                existing = {l.strip() for l in f if l.strip() and not l.strip().startswith("#")}
        to_add = new_ids - existing
        # Only add real S2 IDs (not arxiv: prefixed) to seed files
        to_add = {i for i in to_add if not i.startswith("arxiv:")}
        if to_add:
            with open(filepath, "a", encoding="utf-8") as f:
                for pid in sorted(to_add):
                    f.write(pid + "\n")
        return len(to_add)

    added_pos = _add_ids("seed_paper_positive.csv", fb["liked_ids"])
    added_neg = _add_ids("seed_paper_negative.csv",  fb["disliked_ids"])
    if added_pos or added_neg:
        print(f"  [Refiner] Seeds updated: +{added_pos} positive, +{added_neg} negative from feedback")
