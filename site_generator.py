"""
site_generator.py
-----------------
Builds the static GitHub Pages website from paper data.

Outputs (all at repo root):
  index.html          – today's digest (always overwritten)
  YYYY-MM-DD.html     – permanent archive snapshot for today
  archive.html        – index of all past digests

Uses template.html as the Jinja2 template for daily digest pages.
Archive page is generated inline (no separate template needed).
"""

import os
import glob
import json
import datetime
from datetime import timezone
from jinja2 import Environment, BaseLoader, select_autoescape


# ---------------------------------------------------------------------------
# Load template
# ---------------------------------------------------------------------------

TEMPLATE_FILE = "template.html"

def _make_env() -> Environment:
    """Return a Jinja2 Environment with HTML autoescaping enabled."""
    return Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    )


def _load_template():
    env = _make_env()
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return env.from_string(f.read())


# ---------------------------------------------------------------------------
# Build archive index
# ---------------------------------------------------------------------------

def _build_archive():
    """Scan all YYYY-MM-DD.json files and regenerate archive.html."""
    data_files = sorted(glob.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"), reverse=True)

    entries = []
    for df in data_files:
        date_str = df.replace(".json", "")
        try:
            with open(df, "r", encoding="utf-8") as f:
                d = json.load(f)
            count = len(d.get("papers", []))
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            count = 0
        entries.append({"date": date_str, "count": count, "file": f"{date_str}.html"})

    archive_tmpl = _make_env().from_string(_ARCHIVE_TEMPLATE_SRC)
    html = archive_tmpl.render(entries=entries, build_time=datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    with open("archive.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Site] archive.html updated ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Public build entry point
# ---------------------------------------------------------------------------

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

    # Today's snapshot
    snapshot_file = f"{date_str}.html"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"[Site] {snapshot_file} written")

    # Overwrite index.html
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(rendered)
    print("[Site] index.html updated")

    # Write per-day paper manifest (paperId + title) so the JS feedback widget
    # can initialise the feedback file on first load
    manifest_path = os.path.join("feedback", f"{date_str}_manifest.json")
    manifest = [{"paperId": p.get("paperId",""), "title": p.get("title","")}
                for p in papers if p.get("paperId")]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "papers": manifest}, f, ensure_ascii=False, indent=2)

    # Rebuild archive
    _build_archive()


# ---------------------------------------------------------------------------
# Archive page template (inline — no extra file needed)
# ---------------------------------------------------------------------------

_ARCHIVE_TEMPLATE_SRC = """\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Archive — Daily Paper</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Work+Sans:wght@300..700&display=swap" rel="stylesheet">
<style>
  :root,[data-theme="light"]{--bg:#f7f6f2;--surface:#f9f8f5;--border:#d4d1ca;--text:#28251d;--muted:#7a7974;--primary:#01696f;--primary-h:#0c4e54}
  [data-theme="dark"]{--bg:#171614;--surface:#1c1b19;--border:#393836;--text:#cdccca;--muted:#797876;--primary:#4f98a3;--primary-h:#227f8b}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Work Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100dvh;padding:2rem 1rem}
  .container{max-width:760px;margin:0 auto}
  header{display:flex;justify-content:space-between;align-items:center;margin-bottom:2.5rem;flex-wrap:wrap;gap:1rem}
  h1{font-family:'Instrument Serif',serif;font-size:clamp(1.6rem,4vw,2.4rem)}
  a.back{color:var(--primary);text-decoration:none;font-size:.875rem}
  a.back:hover{color:var(--primary-h)}
  table{width:100%;border-collapse:collapse;font-size:.9rem}
  th{text-align:left;padding:.5rem .75rem;border-bottom:2px solid var(--border);color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
  td{padding:.6rem .75rem;border-bottom:1px solid var(--border)}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:var(--surface)}
  td a{color:var(--primary);text-decoration:none}
  td a:hover{color:var(--primary-h);text-decoration:underline}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.75rem;background:color-mix(in oklab,var(--primary) 15%,var(--surface));color:var(--primary)}
  footer{margin-top:3rem;text-align:center;font-size:.8rem;color:var(--muted)}
  button.theme-toggle{background:none;border:1px solid var(--border);border-radius:6px;padding:.3rem .6rem;cursor:pointer;color:var(--text);font-size:.85rem}
  button.theme-toggle:hover{border-color:var(--primary);color:var(--primary)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <a href="index.html" class="back">← Back to today</a>
      <h1>Archive</h1>
    </div>
    <button class="theme-toggle" data-theme-toggle aria-label="Toggle theme">☀ Light</button>
  </header>
  {% if entries %}
  <table>
    <thead><tr><th>Date</th><th>Papers</th><th>Link</th></tr></thead>
    <tbody>
    {% for e in entries %}
      <tr>
        <td>{{ e.date }}</td>
        <td><span class="badge">{{ e.count }}</span></td>
        <td><a href="{{ e.file }}">View digest →</a></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color:var(--muted);text-align:center;margin-top:4rem">No digests yet. Run the tracker to generate your first one.</p>
  {% endif %}
  <footer>Built {{ build_time }} · <a href="https://github.com" style="color:var(--muted)">TeamDailyPaper</a></footer>
</div>
<script>
(function(){
  const t=document.querySelector('[data-theme-toggle]'),r=document.documentElement;
  let d=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
  r.setAttribute('data-theme',d);
  t&&t.addEventListener('click',()=>{
    d=d==='dark'?'light':'dark';
    r.setAttribute('data-theme',d);
    t.textContent=d==='dark'?'☀ Light':'☾ Dark';
  });
  t&&(t.textContent=d==='dark'?'☀ Light':'☾ Dark');
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        date_arg = sys.argv[1]
        json_file = f"{date_arg}.json"
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            build(date_arg, data.get("papers", []))
        else:
            print(f"No data file found: {json_file}")
    else:
        # Regenerate from today's date if available
        today = datetime.date.today().isoformat()
        json_file = f"{today}.json"
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            build(today, data.get("papers", []))
        else:
            print("Usage: python site_generator.py YYYY-MM-DD")
