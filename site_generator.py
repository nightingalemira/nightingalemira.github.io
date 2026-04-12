"""
site_generator.py
-----------------
Renders index.html, YYYY-MM-DD.html, and archive.html from template.html
using Jinja2. Papers list must already be enriched (with tier, topic_tags, etc.)
"""

import os
import json
import glob
import datetime
from datetime import timezone

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_FILE = "template.html"
ARCHIVE_FILE  = "archive.html"


def _load_template():
    env = Environment(
        loader=FileSystemLoader("."),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template(TEMPLATE_FILE)


def _build_archive():
    data_files = sorted(
        glob.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"),
        reverse=True,
    )
    entries = []
    for path in data_files:
        date_str = path.replace(".json", "")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            count = len(data.get("papers", []))
        except Exception:
            count = 0
        entries.append({"date": date_str, "count": count,
                         "link": f"{date_str}.html"})

    archive_html = _ARCHIVE_TEMPLATE.replace("{{ entries|length }}", str(len(entries)))
    rows = ""
    for e in entries:
        rows += (f'<tr><td>{e["date"]}</td>'
                 f'<td><span class="badge">{e["count"]}</span></td>'
                 f'<td><a href="{e["link"]}">View</a></td></tr>\n')
    archive_html = archive_html.replace("{{ rows }}", rows)

    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        f.write(archive_html)
    print(f"[Site] archive.html updated ({len(entries)} entries)")


def build(date_str: str, papers: list):
    """
    Generate index.html and YYYY-MM-DD.html for the given date and papers list.
    Then regenerate archive.html and per-day paper manifest for feedback.
    """
    os.makedirs("feedback", exist_ok=True)

    tmpl = _load_template()
    rendered = tmpl.render(
        date=date_str,
        papers=papers,
        paper_count=len(papers),
        build_time=datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    snapshot_file = f"{date_str}.html"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"[Site] {snapshot_file} written")

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(rendered)
    print("[Site] index.html updated")

    manifest_path = os.path.join("feedback", f"{date_str}_manifest.json")
    manifest = [{"paperId": p.get("paperId",""), "title": p.get("title","")}
                for p in papers if p.get("paperId")]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "papers": manifest}, f, ensure_ascii=False, indent=2)

    _build_archive()


_ARCHIVE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Archive — Daily Paper</title>
<style>
  :root,[data-theme="light"]{--bg:#f7f6f2;--surface:#f9f8f5;--border:#d4d1ca;--text:#28251d;--muted:#7a7974;--primary:#01696f;--primary-h:#0c4e54}
  [data-theme="dark"]{--bg:#171614;--surface:#1c1b19;--border:#393836;--text:#cdccca;--muted:#797876;--primary:#4f98a3;--primary-h:#227f8b}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Work Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100dvh;padding:2rem}
  .container{max-width:800px;margin:0 auto}
  h1{font-size:1.5rem;margin-bottom:1.5rem;color:var(--text)}
  table{width:100%;border-collapse:collapse;font-size:.9rem}
  th{text-align:left;padding:.5rem .75rem;border-bottom:2px solid var(--border);color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
  td{padding:.6rem .75rem;border-bottom:1px solid var(--border)}
  tr:last-child td{border-bottom:none}
  a{color:var(--primary);text-decoration:none}
  a:hover{color:var(--primary-h)}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.75rem;background:color-mix(in oklab,var(--primary) 15%,var(--surface));color:var(--primary)}
  .back{display:inline-block;margin-bottom:1.5rem;color:var(--muted);font-size:.9rem}
  button.theme-toggle{background:none;border:1px solid var(--border);border-radius:6px;padding:.3rem .6rem;cursor:pointer;color:var(--text);font-size:.85rem}
  button.theme-toggle:hover{border-color:var(--primary);color:var(--primary)}
</style>
</head>
<body>
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem">
    <a href="index.html" class="back">← Back to today</a>
    <button class="theme-toggle" data-theme-toggle aria-label="Toggle theme">☀ Light</button>
  </div>
  <h1>Archive <span class="badge">{{ entries|length }} days</span></h1>
  <table>
    <thead><tr><th>Date</th><th>Papers</th><th>Link</th></tr></thead>
    <tbody>{{ rows }}</tbody>
  </table>
</div>
<script>
(function(){const t=document.querySelector('[data-theme-toggle]'),r=document.documentElement;
let d=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
r.setAttribute('data-theme',d);
t&&t.addEventListener('click',()=>{d=d==='dark'?'light':'dark';r.setAttribute('data-theme',d);
t.textContent=d==='dark'?'☀ Light':'☾ Dark';});})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    today = datetime.date.today().isoformat()
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
        json_path = f"{date_arg}.json"
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            build(date_arg, data.get("papers", []))
        else:
            print(f"ERROR: {json_path} not found")
            sys.exit(1)
    else:
        json_path = f"{today}.json"
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            build(today, data.get("papers", []))
        else:
            build(today, [])
