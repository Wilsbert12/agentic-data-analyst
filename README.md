# Agentic Data Analysis Assistant

A natural language interface for data analysis. Upload one or more CSVs, ask questions in plain English, and receive answers backed by actual data — including exploration, querying, transformation, and business recommendations.

Built as a production-grade agentic application using LangChain, FastAPI, and the Claude API, deployed on AWS with Docker and GitHub Actions CI/CD.

---

## Architecture

```
User (browser)
    ↓
HTML/JS Frontend (chat interface)
    ↓
FastAPI Backend (agent logic)
    ↓
LangChain Agent
    ├── Schema extraction + column samples
    ├── NL → SQL → SQLite query
    ├── LLM interpretation + recommendation
    └── Conversational memory (Chroma vector store)
    ↓
Claude API (LLM)

LangSmith (tracing every agent step)
Docker (frontend + backend containers)
AWS ECR → ECS (cloud deployment)
GitHub Actions (CI/CD — auto-deploy on push)
```

**Key principle:** The LLM never touches raw data. It sees schema and column samples only. Computation runs on the actual data outside the model and results are passed back for interpretation.

---

## Input Types

The agent accepts four types of input to build context for analysis:

| Input | Format | Processing | Destination |
|---|---|---|---|
| Dataset files | CSV | pandas → SQLite | Queryable database |
| Documentation / data card | URL or plain text | Embedded as text | Chroma (RAG) |
| Data dictionary | CSV (column name → description mapping) | Embedded as text | Chroma (RAG) |
| Schema diagrams | Image (PNG, JPG) | Claude vision → text description | Chroma (RAG) |

All non-CSV context ends up in Chroma regardless of input format. Schema images go through a preprocessing step — Claude vision extracts a text description, which is then embedded and stored alongside the other documentation.

**Out of scope / future development:** Formal ERD files (Lucidchart, dbdiagram.io exports). Schema images via Claude vision cover this use case sufficiently for the current scope.

---

## Data Loading — Why pandas?

CSVs are loaded via pandas `read_csv()` rather than directly into SQLite for several reasons:

- **Type inference** — pandas automatically detects integers, floats, strings, and dates. Building this manually would require sampling columns and attempting casts — a worse version of something pandas already does reliably.
- **Bad data handling** — mixed type columns and unconvertible values are handled gracefully via `errors='coerce'`, turning bad values into NaN rather than crashing the load.
- **Simplicity** — `df.to_sql()` creates the SQLite table with correct types already set, replacing manual `CREATE TABLE` and `INSERT` logic entirely.

Each CSV in the `data/` folder is loaded as a separate table in SQLite, using the filename (without extension) as the table name. The agent can query across tables using standard SQL joins.

**Scale note:** pandas loads data into memory — this approach works well for typical analytical datasets but will hit limits on very large files (multi-GB range). In a production environment with datasets of that size, you would skip pandas and query a database directly. For the scope of this project, pandas is the right tool.

---

## Tech Stack

| Component | Tool | Reason |
|---|---|---|
| Frontend | HTML/JS | Clean chat UI, no framework overhead |
| Backend | FastAPI | Industry standard for ML API serving |
| Agent framework | LangChain | Most widely used agentic framework |
| LLM | Claude API | Best available; swappable via config |
| Data layer | SQLite | In-memory, no server, fast for analytics queries |
| Tracing | LangSmith | Native LangChain tracing and evaluation |
| Memory | Chroma | Conversational memory via RAG |
| Containerisation | Docker | Frontend + backend as separate containers |
| Cloud | AWS (ECR + ECS) | ECR for images, ECS for container orchestration |
| CI/CD | GitHub Actions | Auto-deploy on push to main |

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Wilsbert12/agentic-data-analyst.git
cd agentic-data-analyst

# Create and activate virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Add your CSV files to the data/ folder, then run
python main.py
```

---

## API Key

This project requires an Anthropic API key. You can obtain one at [console.anthropic.com](https://console.anthropic.com).

The key is passed through for each session and never stored. If you have concerns about key handling, clone the repo and run it locally — the code is fully transparent.

---

## V2 — Planned

- **Ollama integration** — swap Claude for a locally running open-source model via `.env` config flag, demonstrating LLM provider flexibility without code changes. Known limitation: Claude vision is used in V1 to preprocess schema images into text for Chroma. In Ollama mode this is no longer possible without the Anthropic API. Two options under consideration: (1) drop image input in Ollama mode and document it as a limitation, or (2) integrate a separate open-source vision model alongside Ollama for the preprocessing step. To be decided.
- **Kaggle API integration** — pull datasets directly by providing a Kaggle dataset URL, without manual downloading.
- **Multi-session persistence** — conversation history maintained across sessions, not just within a single session.

---

## Project Status

**Phase 1 — MVP locally (in progress)**
- [x] CSV loading via pandas → SQLite
- [ ] Automatic dataset profiling on upload — shape, dtypes, NaN rates, unique value counts, distributions (matplotlib → base64 → frontend)
- [ ] Schema extraction + column samples passed to agent
- [ ] NL → SQL → answer core loop
- [ ] Simple HTML/JS frontend + FastAPI backend

**Phase 2 — AWS deployment + CI/CD**
- [ ] Dockerfile(s), ECR, ECS, GitHub Actions

**Phase 3 — Feature development**
- [ ] Error handling + retry logic
- [ ] LangSmith tracing
- [ ] NL → pandas transformation layer
- [ ] Recommendation layer
- [ ] Conversational memory via Chroma

**Phase 4 — Polish and documentation**
- [ ] README, architecture diagram, demo GIF
- [ ] Portfolio write-up