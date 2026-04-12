# TeamDailyPaper

A GitHub Pages-hosted daily literature review website powered by LLMs.  
Runs automatically every day via GitHub Actions — papers are fetched, summarised, and published to your team's `*.github.io` site.

---

## Quick Start

1. **Fork / clone this repo** into your GitHub organisation.
2. **Configure GitHub Pages**:  
   Repo Settings → Pages → Source: **Deploy from branch** → Branch: `main` / `(root)`
3. **Add GitHub Secrets** (Settings → Secrets and variables → Actions):

   | Secret | Required | Description |
   |---|---|---|
   | `OPENROUTER_API_KEY` | ✅ (if using OpenRouter) | [openrouter.ai](https://openrouter.ai) key |
   | `KILO_API_KEY` | ✅ (if using KiloCode) | [kilo.codes](https://kilo.codes) key |
   | `OLLAMA_BASE_URL` | ✅ (if using Ollama) | e.g. `http://your-server:11434/v1` |
   | `S2_API_KEY` | optional | Semantic Scholar key (higher rate limit) |
   | `LLM_PROVIDER` | optional | Override provider: `openrouter` / `ollama` / `kilo` |
   | `LLM_MODEL` | optional | Override model name |

4. **Copy the workflow file**:
   ```bash
   mkdir -p .github/workflows
   cp daily_update.yml .github/workflows/daily_update.yml
   ```
5. **Edit your config files** (see below).
6. Push to `main` — the first run will trigger automatically at 07:00 UTC, or you can trigger it manually from the Actions tab.

Your site will be live at `https://<your-org>.github.io/<repo-name>/`

---

## Configuration

All config lives at the repo root — no subdirectories.

### `llm_config.yaml` — LLM Provider

```yaml
provider: openrouter      # openrouter | ollama | kilo
model: ""                 # blank = use default free model for provider
base_url: ""              # blank = use standard URL for provider
temperature: 0.3
max_tokens: 1024
```

**Default free models** (used when `model` is blank):

| Provider | Default model |
|---|---|
| `openrouter` | `meta-llama/llama-4-maverick:free` |
| `ollama` | `llama3.3:70b` |
| `kilo` | `moonshotai/moonshot-v1-8k` |

---

### `query_config.yaml` — Paper Retrieval

```yaml
mode: similarity          # similarity | query
```

**Mode A — Similarity** (`mode: similarity`):  
Uses the [Semantic Scholar Recommendations API](https://api.semanticscholar.org/api-docs/#tag/Paper-Recommendations).  
Add paper IDs to `seed_paper_positive.csv` (papers you like) and optionally `seed_paper_negative.csv` (papers you dislike).

**Mode B — Query** (`mode: query`):  
Searches Semantic Scholar and/or arXiv using keywords you specify.

```yaml
mode: query
keywords:
  - large language models
  - retrieval augmented generation
sources:
  - semanticscholar
  - arxiv
max_papers: 10
```

**Custom LLM prompts** (both modes):  
Override the built-in summarisation prompts by setting `system_prompt` and `user_prompt`.  
Available variables in `user_prompt`: `{title}`, `{tldr}`, `{abstract}`.

---

### `seed_paper_positive.csv`

One Semantic Scholar paper ID per line. Find IDs at [semanticscholar.org](https://www.semanticscholar.org).

### `blacklisted_venues.txt`

Venue name substrings to exclude (case-insensitive), one per line.

---

## Repository Files

| File | Purpose |
|---|---|
| `paper_tracker.py` | Main entry point — fetch → summarise → write JSON → call site generator |
| `llm_adapter.py` | Multi-provider LLM factory (Ollama, OpenRouter, KiloCode) |
| `site_generator.py` | Renders `index.html`, `YYYY-MM-DD.html`, `archive.html` from Jinja2 template |
| `template.html` | Jinja2 HTML template for daily digest pages |
| `daily_update.yml` | GitHub Actions workflow (copy to `.github/workflows/`) |
| `llm_config.yaml` | LLM provider and model settings |
| `query_config.yaml` | Paper retrieval mode, keywords, custom prompts |
| `seed_paper_positive.csv` | S2 seed paper IDs (positive) |
| `seed_paper_negative.csv` | S2 seed paper IDs (negative) |
| `blacklisted_venues.txt` | Venue blacklist |
| `seen_papers.txt` | Auto-updated dedup history (committed by bot) |
| `requirements.txt` | Python dependencies |
| `index.html` | Generated: today's digest |
| `archive.html` | Generated: archive of all past digests |
| `YYYY-MM-DD.html` | Generated: permanent daily snapshot |
| `YYYY-MM-DD.json` | Generated: raw paper data for that date |

---

## Running Locally

```bash
pip install -r requirements.txt

# Set your API keys
export OPENROUTER_API_KEY=sk-...
export S2_API_KEY=...          # optional

# Run the full pipeline
python paper_tracker.py

# Regenerate site from existing data only (no API calls)
python site_generator.py 2026-04-10
```

---

## Switching LLM Providers

**OpenRouter (default)**
```yaml
# llm_config.yaml
provider: openrouter
model: ""   # uses meta-llama/llama-4-maverick:free
```
Set secret: `OPENROUTER_API_KEY`

**Ollama (local)**
```yaml
provider: ollama
model: llama3.3:70b
base_url: ""   # uses http://localhost:11434/v1 by default
```
Set secret: `OLLAMA_BASE_URL` (if your Ollama is on a remote server)

**KiloCode**
```yaml
provider: kilo
model: ""
```
Set secret: `KILO_API_KEY`

---

## License

MIT
