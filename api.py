"""
api.py — FastAPI backend for the Agentic Data Analysis Assistant.

Endpoints are organised into four groups matching the user flow:
  1. Session management  — create, persist, restore, and clear sessions
  2. Setup phase         — upload files, build schema context, confirm with Claude
  3. Analysis phase      — outer loop (new question) and inner loop (follow-up)
  4. Memory              — summarise and store completed analyses in Chroma

All session state lives in the in-memory `sessions` dict and is persisted to
sessions/{session_id}/session.json after every state-changing operation, so the
server can recover from restarts without losing active sessions.
"""

import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

import chromadb
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from utils import get_claude_client, build_setup_messages

app = FastAPI()


# Convert all unhandled exceptions to JSON so the frontend can parse the error
# message — without this, FastAPI returns plain-text "Internal Server Error".
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SESSION STORE ─────────────────────────────────────────────────────────────

# In-memory session store. Each entry holds the full session state:
# api_key, messages, system_prompt, analysis_count, chroma_collection, agent.
sessions = {}


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying up to 3 times on a 429 rate limit error.

    Waits 62 seconds between attempts — long enough for Anthropic's per-minute
    token limit to reset. On any other exception, re-raises immediately.
    """
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < 2 and ("429" in str(e) or "rate_limit" in str(e).lower()):
                time.sleep(62)
                continue
            raise


def _session_path(session_id: str) -> str:
    """Return the path to the session's JSON persistence file."""
    return f"sessions/{session_id}/session.json"


def _persist_session(session_id: str):
    """Write serialisable session fields to disk.

    Saves api_key, analysis_count, messages, and system_prompt. Non-serialisable
    objects (chroma_collection, agent) are excluded and reconstructed on restore.
    Silently swallows write errors — persistence is best-effort.
    """
    session = sessions.get(session_id, {})
    os.makedirs(f"sessions/{session_id}", exist_ok=True)
    data = {
        "api_key": session.get("api_key", ""),
        "analysis_count": session.get("analysis_count", 0),
        "messages": session.get("messages", []),
        "system_prompt": session.get("system_prompt", ""),
    }
    try:
        with open(_session_path(session_id), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _restore_session(session_id: str) -> dict | None:
    """Attempt to reconstruct a session from disk after a server restart.

    Reads session.json for serialisable fields, then reconnects to the Chroma
    collection if the chroma_db directory exists. Returns None if no persisted
    data is found (i.e. the session never existed).
    """
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    chroma_path = f"sessions/{session_id}/chroma_db"
    if os.path.isdir(chroma_path):
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            data["chroma_collection"] = client.get_collection(name="schema_context")
        except Exception:
            pass
    return data


def get_session(session_id: str) -> dict:
    """Return the session dict for session_id.

    Checks the in-memory store first. On a miss, attempts to restore from disk
    (handles server restarts). Raises 404 if the session cannot be found or
    reconstructed.
    """
    if session_id not in sessions:
        restored = _restore_session(session_id)
        if restored is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Please refresh and start a new session.",
            )
        sessions[session_id] = restored
    return sessions[session_id]


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    """Used by /create-session — carries the user's Anthropic API key."""
    api_key: str


class SessionAction(BaseModel):
    """Used by endpoints that only need a session_id."""
    session_id: str


class UserMessage(BaseModel):
    """Used by endpoints that need a session_id and a user message."""
    session_id: str
    user_input: str


# ── SESSION MANAGEMENT ────────────────────────────────────────────────────────

@app.post("/create-session")
def create_session(request: SessionRequest):
    """Create a new session and persist it to disk immediately."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "api_key": request.api_key,
        "messages": [],
        "analysis_count": 0
    }
    _persist_session(session_id)
    return {"session_id": session_id}


@app.post("/clear-memory")
def clear_memory(request: SessionAction):
    """Delete analysis summaries from Chroma, wipe the session folder, and remove
    the session from memory. Used when the user wants a clean slate for a single session."""
    session = get_session(request.session_id)
    if "chroma_collection" in session:
        existing = session["chroma_collection"].get()
        ids_to_delete = [id for id in existing['ids'] if id.startswith('analysis_')]
        if ids_to_delete:
            session["chroma_collection"].delete(ids=ids_to_delete)
    shutil.rmtree(f"sessions/{request.session_id}", ignore_errors=True)
    sessions.pop(request.session_id, None)
    return {"status": "ok", "message": "Session cleared"}


@app.post("/clear-all-sessions")
def clear_all_sessions():
    """Wipe the entire sessions/ directory and clear the in-memory store.
    Used by the 'Clear Memory' button in the frontend to reset everything."""
    shutil.rmtree("sessions", ignore_errors=True)
    os.makedirs("sessions", exist_ok=True)
    sessions.clear()
    return {"status": "ok", "message": "All sessions cleared"}


# ── SETUP PHASE ───────────────────────────────────────────────────────────────

@app.post("/load-data")
async def load_data(session_id: str = Form(...), files: list[UploadFile] = None):
    """Save uploaded CSVs to disk and load them into the session's SQLite database.

    Each CSV becomes a table named after the file stem. Uses multipart/form-data
    so files and the session_id can be sent together without a JSON body.
    """
    os.makedirs(f"sessions/{session_id}/data", exist_ok=True)
    connection = sqlite3.connect(f"sessions/{session_id}/database.sqlite")
    for file in files:
        path = f"sessions/{session_id}/data/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    for file in Path(f"sessions/{session_id}/data/").glob("*.csv"):
        df = pd.read_csv(file)
        df.to_sql(file.stem, connection, if_exists="replace", index=False)
    connection.commit()
    connection.close()
    return {"message": "Data loaded successfully"}


@app.post("/load-context")
async def load_context(session_id: str = Form(...), files: list[UploadFile] = None):
    """Save uploaded context files (text, images) to the session's context/ folder.

    These are read by build_setup_messages() and sent to Claude during setup.
    Uses multipart/form-data for the same reason as /load-data.
    """
    os.makedirs(f"sessions/{session_id}/context", exist_ok=True)
    for file in files:
        path = f"sessions/{session_id}/context/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"message": "Context files loaded successfully"}


@app.post("/build-setup-messages")
def build_setup_messages_endpoint(request: SessionAction):
    """Build the initial Claude message list from uploaded files and store it in
    the session. Called once before /initial-prompt."""
    messages = build_setup_messages(request.session_id)
    get_session(request.session_id)["messages"] = messages
    return {"status": "ok"}


@app.post("/initial-prompt")
def initial_prompt(request: SessionAction):
    """Send the schema data to Claude and receive its initial analysis and questions.

    Stores the system prompt in the session so /setup-reply can reuse it across
    the schema confirmation loop.
    """
    system_prompt = """
    You are a data analysis assistant. You have been provided with schema information,
    sample rows, and profiling statistics for a dataset consisting of one or more tables.

    Your first task is to fully understand the dataset before any analysis begins.
    To do this:

    1. Review all provided table schemas, sample rows, and profiling statistics
    2. Produce a concise summary of what the dataset represents
    3. Identify and list all primary keys, foreign keys, and relationships between tables in a clear table format
    4. Flag any ambiguities or questions about the schema that need clarification
    5. Ask those questions and wait for confirmation before proceeding
    6. Repeat steps 4 and 5 until all open questions are resolved
    7. Once all questions are clarified, output exactly: [SCHEMA CONFIRMED]

    Do not begin any analysis until the schema is fully understood and confirmed.
    """
    session = get_session(request.session_id)
    session["system_prompt"] = system_prompt
    client = get_claude_client(api_key=session["api_key"])
    response = _with_retry(client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system_prompt,
        messages=session["messages"]
    )
    session["messages"].append({
        "role": "assistant",
        "content": response.content[0].text
    })
    return {"response": response.content[0].text}


@app.post("/setup-reply")
def setup_reply(request: UserMessage):
    """Continue the schema confirmation loop with the user's reply.

    Returns confirmed=True when Claude's response contains [SCHEMA CONFIRMED],
    which signals the frontend to proceed to finishSetup().
    """
    session = get_session(request.session_id)
    session["messages"].append({"role": "user", "content": request.user_input})
    client = get_claude_client(api_key=session["api_key"])
    response = _with_retry(client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=session["system_prompt"],
        messages=session["messages"]
    )
    session["messages"].append({"role": "assistant", "content": response.content[0].text})
    confirmed = "[SCHEMA CONFIRMED]" in response.content[0].text
    return {
        "response": response.content[0].text,
        "confirmed": confirmed
    }


@app.post("/save-setup-convo-to-json")
def save_setup_to_json(request: SessionAction):
    """Persist the setup conversation (messages[1:]) to disk as JSON.

    Skips messages[0] because it contains the raw schema data (large CSV samples
    and profiling stats) which don't need to be stored long-term. The JSON is
    used later as the source of schema context in /analysis-outer-loop.
    """
    session = get_session(request.session_id)
    with open(f"sessions/{request.session_id}/setup_conversation.json", "w") as f:
        json.dump(session["messages"][1:], f, indent=2)
    return {"status": "ok"}


@app.post("/save-setup-convo-to-chroma")
def save_setup_to_chroma(request: SessionAction):
    """Store the setup conversation as a single document in the Chroma vector store.

    Tagged with type='setup' so it can be excluded from RAG retrieval during
    analysis (only past analysis summaries are retrieved, not the setup itself).
    Also stores the collection object in the session for later use.
    """
    session = get_session(request.session_id)
    conversation_text = ""
    for message in session["messages"][1:]:
        conversation_text += f"{message['role']}: {message['content']}\n\n"
    chroma_client = chromadb.PersistentClient(path=f"sessions/{request.session_id}/chroma_db")
    collection = chroma_client.get_or_create_collection(name="schema_context")
    collection.upsert(
        documents=[conversation_text],
        ids=["setup_conversation"],
        metadatas=[{"type": "setup"}]
    )
    session["chroma_collection"] = collection
    return {"status": "ok"}


@app.post("/initialise-agent")
def initialise_agent(request: SessionAction):
    """Create the LangGraph agent for the analysis phase.

    run_sql is defined as a closure so it captures the session-specific database
    path — the docstring is required by create_agent to generate a tool description.
    A new agent is created per session so each session gets its own database connection.
    """
    session = get_session(request.session_id)
    db_path = f"sessions/{request.session_id}/database.sqlite"

    def run_sql(query: str) -> str:
        """Execute a SQL query against the dataset and return the results as a string."""
        conn = sqlite3.connect(db_path)
        try:
            result = conn.execute(query).fetchall()
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            conn.close()

    llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=session["api_key"])
    agent = create_agent(model=llm, tools=[run_sql])
    session["agent"] = agent
    return {"status": "ok"}


# ── ANALYSIS PHASE ────────────────────────────────────────────────────────────

@app.post("/analysis-outer-loop")
def analysis_outer_loop(request: UserMessage):
    """Start a new analysis for a fresh business question.

    Builds the system prompt by injecting:
      - The confirmed schema summary (last assistant message from setup JSON)
      - Up to 2 relevant past analyses retrieved from Chroma via similarity search

    The system prompt is marked with cache_control=ephemeral so Anthropic caches
    it server-side — cached tokens don't count against the 30K TPM rate limit.
    Resets the message history for the new analysis.
    """
    session = get_session(request.session_id)

    # Inject schema context from the setup conversation (last assistant message only,
    # to keep token count low while retaining the confirmed schema summary)
    with open(f"sessions/{request.session_id}/setup_conversation.json") as f:
        schema_context = json.load(f)
    assistant_msgs = [m for m in schema_context if m["role"] == "assistant"]
    schema_text = assistant_msgs[-1]["content"] if assistant_msgs else ""

    # Retrieve relevant past analyses from Chroma (excludes the setup document)
    results = session["chroma_collection"].query(
        query_texts=[request.user_input],
        n_results=2,
        where={"type": {"$ne": "setup"}}
    )
    context_past_conversations = "\n\n".join(results['documents'][0])

    system_prompt = f"""
    You are an expert data analyst. You have full knowledge of the dataset schema and confirmed handling rules from the setup phase:

    {schema_text}

    Additional context from past analyses, if available:
    {context_past_conversations}

    When the user asks a business question, follow these steps:

    1. Before doing anything else, assess whether the question is specific enough to answer correctly. If it is ambiguous, vague, or could be interpreted in multiple ways, ask clarifying questions. Do not assume criteria or definitions — always ask.
    2. Once the question is clear, identify what data is needed and how to retrieve it via SQL.
    3. If the answer requires multiple queries, break it down into clear steps and walk the user through your plan before executing.
    4. Execute the necessary queries.
    5. Present the results in plain language that directly answers the original business question. Be precise about what the data actually represents — do not conflate an attribute of a record with the record itself.

    Always ground your answers in actual query results — never guess or estimate.
    Always ask before assuming criteria, thresholds, or definitions that are not explicitly stated in the question.

    If the schema context flags anything as an open data quality issue or marks something as needing investigation, do not assume a handling approach. Ask the user how to handle it before writing any query that touches that data.

    When you have completed an analysis with actual query results, end your response with [ANALYSIS COMPLETE].
    Do not include this marker when asking clarifying questions.
    """
    session["system_prompt"] = system_prompt
    session["messages"] = []
    cached_system = SystemMessage(content=[{
        "type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}
    }])
    response = _with_retry(
        session["agent"].invoke,
        {"messages": [cached_system, {"role": "user", "content": request.user_input}]}
    )
    session["messages"].append({"role": "user", "content": request.user_input})
    session["messages"].append({"role": "assistant", "content": response["messages"][-1].content})
    _persist_session(request.session_id)
    satisfied = "[ANALYSIS COMPLETE]" in response["messages"][-1].content
    return {"response": response["messages"][-1].content, "satisfied": satisfied}


@app.post("/analysis-inner-loop")
def analysis_inner_loop(request: UserMessage):
    """Continue refining the current analysis with a follow-up from the user.

    Passes the full conversation history of the current analysis on every call —
    all prior turns within the current question are relevant to the refinement.
    Uses the same cached system prompt as the outer loop.
    """
    session = get_session(request.session_id)
    session["messages"].append({"role": "user", "content": request.user_input})
    cached_system = SystemMessage(content=[{
        "type": "text", "text": session["system_prompt"], "cache_control": {"type": "ephemeral"}
    }])
    response = _with_retry(
        session["agent"].invoke,
        {"messages": [cached_system] + session["messages"]}
    )
    session["messages"].append({"role": "assistant", "content": response["messages"][-1].content})
    _persist_session(request.session_id)
    satisfied = "[ANALYSIS COMPLETE]" in response["messages"][-1].content
    return {"response": response["messages"][-1].content, "satisfied": satisfied}


# ── MEMORY ────────────────────────────────────────────────────────────────────

@app.post("/add-user-msg-to-inner-loop")
def add_user_msg_to_inner_loop(request: SessionAction):
    """Append a summarisation request to the current analysis messages.

    Called immediately before /analysis-summary so that Claude is asked to
    summarise the conversation it has just seen.
    """
    session = get_session(request.session_id)
    session["messages"].append({
        "role": "user",
        "content": "Please summarise the above analysis concisely for future reference."
    })
    _persist_session(request.session_id)
    return {"status": "ok"}


@app.post("/analysis-summary")
def analysis_summary(request: SessionAction):
    """Summarise the completed analysis and store it in Chroma for future RAG retrieval.

    Only called when the user confirms they are satisfied with the answer. The
    summarisation prompt strips out misunderstandings, dead ends, and clarifying
    exchanges — only the confirmed findings and SQL logic are stored.
    Increments analysis_count so each summary gets a unique Chroma document ID.
    """
    summarisation_prompt = """
    Summarise the following data analysis conversation for storage in a retrieval system.

    Include only:
    - The confirmed business question that was answered
    - The key analytical findings and conclusions
    - The SQL logic or approach that produced the correct results
    - Any important data quality notes or caveats that affected the analysis

    Exclude:
    - Misunderstandings or incorrect answers that were corrected
    - Dead ends or failed approaches
    - Back-and-forth clarification exchanges
    - Any part of the conversation that did not contribute to the final answer
    """
    session = get_session(request.session_id)
    client = get_claude_client(api_key=session["api_key"])
    summary = _with_retry(client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=summarisation_prompt,
        messages=session["messages"]
    )
    summary_text = summary.content[0].text
    session["chroma_collection"].upsert(
        documents=[summary_text],
        ids=[f'analysis_{session["analysis_count"]}'],
        metadatas=[{"type": "analysis"}]
    )
    session["analysis_count"] += 1
    _persist_session(request.session_id)
    return {"status": "ok"}


# Static files must be mounted last — it acts as a catch-all for any path not
# matched by the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
