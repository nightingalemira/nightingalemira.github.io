"""
prompt_refiner.py — keyword refinement and feedback sync.
Updates ONLY keywords in query_config.yaml; never overwrites prompts.
"""

import os, json, re, glob, yaml
from llm_adapter import get_client, chat

FEEDBACK_DIR      = "feedback"
QUERY_CONFIG_FILE = "query_config.yaml"

REFINE_SYSTEM = """\
You are a research librarian. The user describes their research interest.
Distil it into 3-7 short search phrases (each ≤5 words) for academic paper APIs.
Output ONLY a JSON object with one key: "keywords" (list of strings).
No extra keys, no markdown fences, no preamble."""

def refine_interest_to_prompts(interest: str, client=None, cfg=None) -> dict:
    if client is None or cfg is None:
        client, cfg = get_client()
    from paper_tracker import _safe_format, _extract_json
    user_msg = _safe_format("User research interest:\n{interest}", interest=interest)
    try:
        raw    = chat(REFINE_SYSTEM, user_msg, client=client, cfg=cfg)
        result = _extract_json(raw)
        if not isinstance(result.get("keywords"), list) or not result["keywords"]:
            raise ValueError("keywords missing or empty")
        return {"keywords": [str(k).strip() for k in result["keywords"] if str(k).strip()]}
    except Exception as exc:
        print(f"  [Refiner] LLM refinement failed ({exc}); using naive extraction")
        stop = {"a","an","the","and","or","in","on","for","of","with","is","are","we","i"}
        words = [w for w in re.findall(r"[a-zA-Z]{3,}", interest.lower()) if w not in stop]
        phrases = [" ".join(words[i:i+3]) for i in range(len(words)-2)][:6] or [interest[:50]]
        return {"keywords": phrases}


def load_all_feedback() -> dict:
    liked, disliked, liked_ids, disliked_ids = [], [], set(), set()
    if not os.path.exists(FEEDBACK_DIR):
        return {"liked_titles":[],"disliked_titles":[],"liked_ids":set(),"disliked_ids":set()}
    for fpath in sorted(glob.glob(os.path.join(FEEDBACK_DIR,"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("feedback", []):
                pid, title, vote = entry.get("paperId",""), entry.get("title",""), entry.get("vote","")
                if vote == "up":   liked_ids.add(pid); liked.append(title) if title else None
                elif vote == "down": disliked_ids.add(pid); disliked.append(title) if title else None
        except Exception:
            continue
    return {"liked_titles":liked[-30:],"disliked_titles":disliked[-30:],
            "liked_ids":liked_ids,"disliked_ids":disliked_ids}


def maybe_refresh_prompts(client=None, cfg_llm=None) -> bool:
    """Refresh ONLY keywords based on feedback; never touches prompts."""
    if client is None or cfg_llm is None:
        client, cfg_llm = get_client()
    if not os.path.exists(QUERY_CONFIG_FILE):
        return False
    with open(QUERY_CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("mode") != "query" or not cfg.get("user_interest"):
        return False
    fb = load_all_feedback()
    if not fb["liked_titles"] and not fb["disliked_titles"]:
        print("  [Refiner] No feedback yet; skipping refresh")
        return False

    from paper_tracker import _extract_json
    liked_str    = "\n".join(f"  + {t}" for t in fb["liked_titles"][-15:]) or "  (none)"
    disliked_str = "\n".join(f"  - {t}" for t in fb["disliked_titles"][-15:]) or "  (none)"
    refresh_sys  = ("Update search keywords based on feedback. "
                    "Output ONLY JSON: {\"keywords\": [...], \"changed\": bool}. No other keys.")
    refresh_user = (f"Current keywords: {cfg.get('keywords', [])}\n"
                    f"Liked papers:\n{liked_str}\nDisliked papers:\n{disliked_str}")
    try:
        raw    = chat(refresh_sys, refresh_user, client=client, cfg=cfg_llm)
        result = _extract_json(raw)
    except Exception as exc:
        print(f"  [Refiner] Refresh failed: {exc}"); return False

    if not result.get("changed", False):
        print("  [Refiner] No change needed"); return False
    new_kw = result.get("keywords", cfg.get("keywords", []))
    if not isinstance(new_kw, list) or not new_kw:
        return False
    # Update only keywords
    cfg["keywords"] = [str(k).strip() for k in new_kw if str(k).strip()]
    with open(QUERY_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [Refiner] Keywords updated: {cfg['keywords']}")
    return True


def sync_feedback_to_seeds():
    """Add liked papers to positive seeds, disliked to negative seeds."""
    import csv
    fb = load_all_feedback()
    for seed_file, ids in [("seed_paper_positive.csv", fb["liked_ids"]),
                            ("seed_paper_negative.csv", fb["disliked_ids"])]:
        if not ids:
            continue
        existing = set()
        if os.path.exists(seed_file):
            with open(seed_file, newline="", encoding="utf-8") as f:
                existing = {row[0].strip() for row in csv.reader(f) if row}
        new_ids = ids - existing
        if new_ids:
            with open(seed_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for pid in sorted(new_ids):
                    w.writerow([pid])
            print(f"  [Seeds] Added {len(new_ids)} IDs to {seed_file}")
