"""
utils.py — Helper functions for session setup and Claude client initialisation.

These are called by api.py during the setup phase. Nothing here is request-aware —
all functions take plain arguments and return plain values.
"""

import base64
import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic


# ── DATA PROFILING ────────────────────────────────────────────────────────────

def basic_analysis(df) -> str:
    """Return a plain-text profiling summary for a single DataFrame.

    Includes shape, column types, null rates, unique value counts, and
    descriptive statistics. Used during setup to give Claude a statistical
    overview of each table before schema confirmation.
    """
    return (str(df.shape) + "\n\n" +
            df.dtypes.to_string() + "\n\n" +
            df.isnull().mean().to_string() + "\n\n" +
            df.nunique().to_string() + "\n\n" +
            df.describe().to_string())


# ── SETUP MESSAGE BUILDER ─────────────────────────────────────────────────────

def build_setup_messages(session_id: str) -> list:
    """Build the initial message list to send to Claude for schema confirmation.

    Reads all files from the session's context/ and data/ directories and
    assembles them into a single Anthropic-format user message:
      - context/: text files and CSVs are embedded as text blocks;
                  images are base64-encoded and sent as image blocks.
      - data/:    for each CSV, sends 5 sample rows (via SQLite) and a
                  profiling summary (via pandas).

    Raises ValueError if no content is found — this means no files were
    uploaded before setup was triggered.
    """
    messages = [{"role": "user"}]
    content = []

    # --- Context files (documentation, data cards, schema images) ---
    context_path = Path(f"sessions/{session_id}/context/")
    for file in (context_path.iterdir() if context_path.exists() else []):
        if file.suffix == '.txt':
            with open(file) as f:
                content.append({"type": "text", "text": f.read()})
        elif file.suffix == '.csv':
            content.append({"type": "text", "text": pd.read_csv(file).to_string()})
        elif file.suffix in ['.jpg', '.jpeg', '.png']:
            media_type = "image/jpeg" if file.suffix in ['.jpg', '.jpeg'] else "image/png"
            with open(file, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": file_content}
                })

    # --- Dataset files (sample rows + profiling stats per table) ---
    for file in Path(f"sessions/{session_id}/data/").glob("*.csv"):
        connection = sqlite3.connect(f"sessions/{session_id}/database.sqlite")
        df_sample = pd.read_sql(f"SELECT * FROM {file.stem} LIMIT 5", connection)
        connection.close()
        df = pd.read_csv(file)
        content.append({"type": "text", "text": f"Sample rows from table '{file.stem}' (5 rows, all columns):"})
        content.append({"type": "text", "text": df_sample.to_string(max_cols=None, line_width=None)})
        content.append({"type": "text", "text": f"Basic profiling for table '{file.stem}':"})
        content.append({"type": "text", "text": basic_analysis(df)})

    if not content:
        raise ValueError(f"No data or context files found for session '{session_id}'. Upload at least one CSV before building setup messages.")

    messages[0]["content"] = content
    return messages


# ── CLAUDE CLIENT ─────────────────────────────────────────────────────────────

def get_claude_client(api_key: str = None) -> anthropic.Anthropic:
    """Return an Anthropic client.

    If api_key is provided (from the session, entered by the user in the UI),
    uses that. Otherwise falls back to ANTHROPIC_API_KEY in .env — used for
    local development only.
    """
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    load_dotenv()
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
