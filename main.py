# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================

import sqlite3
from pathlib import Path
import pandas as pd
import base64
import os
from dotenv import load_dotenv
import anthropic
import json
import chromadb
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

# =============================================================================
# 2. SETUP PHASE
# =============================================================================

# Deleting Memory from earlier sessions
@app.post("/clear-memory")
def clear_memory(session_id: str):
    existing = collection.get()
    ids_to_delete = [id for id in existing['ids'] if id.startswith('analysis_')]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        return {"message": f"Cleared {len(ids_to_delete)} analyses from memory"}


# Load each CSV into pandas then into SQLite
@app.post("/load-data")
async def load_data(session_id: str, files: list[UploadFile]):
    os.makedirs(f"sessions/{session_id}/data", exist_ok=True)
    connection = sqlite3.connect(f"sessions/{session_id}/database.sqlite")
    for file in files:
        path = f"sessions/{session_id}/data/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    for file in Path(f"sessions/{session_id}/data/").glob("*.csv"):
        df = pd.read_csv(file)
        df.to_sql(file.stem, connection, if_exists="replace", index=False)
    return {"message": "Data loaded successfully"}

# Loading additional file for context (seperate upload area in frontend)
@app.post("/load-context")
async def load_context(session_id: str, files: list[UploadFile]):
    for file in files:
        path = f"sessions/{session_id}/context/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"message": "Context files loaded successfully"}

# Build context message for Claude API from files in the context/ folder
# Supports: .txt (data cards), .csv (data dictionaries), .jpg/.png (schema images via vision)

# Function to return basic analytics as a string to pass to llm via function build_setup_messages(session_id)
def basic_analysis (df):
    return (str(df.shape) + "\n\n" + 
            df.dtypes.to_string() + "\n\n" + 
            df.isnull().mean().to_string() + "\n\n" + 
            df.nunique().to_string() + "\n\n" + 
            df.describe().to_string())

# Setting up the message for the initial prompt
@app.post("/setup-message")
def build_setup_messages(session_id) -> list:
    messages=[{"role": "user"}]
    content=[] 
    for file in Path(f"sessions/{session_id}/context/").iterdir():
        if file.suffix == '.txt':
            file_content = open(file).read()
            content.append(
                {
                    "type": "text",
                    "text": file_content
                }
                )
        elif file.suffix == '.csv':
            file_content = pd.read_csv(file)
            file_content = file_content.to_string()
            content.append(
                {
                    "type": "text",
                    "text": file_content
                }
                )
        elif file.suffix in ['.jpg', '.png']:
            with open(file, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode()
                content.append(
                {
                    "type":"image",
                    "source": {
                        "type": "base64",
                        "media_type": "file.suffix",
                        "data": file_content
                    }
                }
                )
        else: pass

    # Adding 5 sample rows with explanatory intro for context
    for file in Path(f"sessions/{session_id}/data/").glob("*.csv"):
        content.append(
            {
            "type": "text",
            "text": f"Sample rows from table '{file.stem}' (5 rows, all columns):"
            })
        connection = sqlite3.connect(f"sessions/{session_id}/database.sqlite")
        df_sample = pd.read_sql(f"SELECT * FROM {file.stem} LIMIT 5", connection)
        file_content = df_sample.to_string(max_cols=None, line_width=None)
        content.append(
            {
                "type": "text",
                "text": file_content
            })
        df = pd.read_csv(file)
        # Adding the basic analytics
        content.append(
            {
            "type": "text",
            "text": f"Basic profiling for table '{file.stem}':"
            })
        content.append(
                {
                    "type": "text",
                    "text": basic_analysis (df)
                }
                )
    messages[0]["content"] = content
    return messages

# Creating a session with the api_key and assigning session ID
@app.post("/create-session")
def create_session(api_key: str):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"api_key": api_key}
    return {"session_id": session_id}

# Helper function to get Claude client (If main is called manually retrive form .env, if key is provided use key)
def get_claude_client(api_key: str = None):
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    else:
        load_dotenv()
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
# Sending intial promt (incl. sys prompt) to llm
@app.post("/initial-prompt")
def initial_prompt(session_id) -> str:
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
    # First response from Claude
    #return {message: "Analysing data..."}
    client = get_claude_client(api_key=sessions[session_id]["api_key"])
    response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system_prompt,
            messages=sessions[session_id]["messages"])
    sessions[session_id]["messages"].append({
        "role": "assistant",
        "content": response.content[0].text
    })
    return {"response": response.content[0].text}

# Conversational loop, until Claude understands data
@app.post("/setup-reply")
def setup_reply(session_id: str, user_input: str):
    # append user input to messages
    sessions[session_id]["messages"].append({"role": "user", "content": user_input})
    
    # call Claude
    client = get_claude_client(api_key=sessions[session_id]["api_key"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=sessions[session_id]["system_prompt"],
        messages=sessions[session_id]["messages"]
    )

    # append Claude response to messages
    sessions[session_id]["messages"].append({"role": "assistant", "content": response.content[0].text})
    
    # check if setup is complete
    confirmed = "[SCHEMA CONFIRMED]" in response.content[0].text
    
    return {
        "response": response.content[0].text,
        "confirmed": confirmed
    }

# Save full conversation (except initial input) to file for reference. Used to bypass Chroma (to always give llm setup context)
@app.post("/save-setup-convo-to-json")
def save_setup_to_json(session_id: str):
    with open(f"sessions/{session_id}/setup_conversation.json", "w") as f:
        json.dump(sessions[session_id]["messages"][1:], f, indent=2)
    return {"status": "ok"}
    

# After all questions are resolved and conversation is ended
# Save the conversation except initial user input (text, images etc.) into a string
@app.post("/save-setup-convo-to-chroma")
def save_setup_to_chroma(session_id: str): 
    conversation_text = ""
    for message in sessions[session_id]["messages"][1:]:
        role = message["role"]
        content = message["content"]
        conversation_text += f"{role}: {content}\n\n"
    # Create and connect to Chroma database (RAG)
    # Store confirmed schema understanding in Chroma
    chroma_client = chromadb.PersistentClient(path=f"sessions/{session_id}/chroma_db")
    collection = chroma_client.get_or_create_collection(name="schema_context")
    collection.upsert(
        documents=[conversation_text],
        ids=["setup_conversation"],
        metadatas=[{"type": "setup"}]
    )
    sessions[session_id]["chroma_collection"] = collection
    return {"status": "ok"}


# =============================================================================
# 3. ANALYTICAL PHASE — Core Loop
# =============================================================================


# Building an SQL Tool for the agent to use (function)
def run_sql(query: str) -> str:
    """Execute a SQL query against the SQLite database and return the results."""
    conn = sqlite3.connect("sessions/database.sqlite")
    try:
        result = conn.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()


# Setting up the agent once before the core loop
@app.post("/initialise-agent")
def initialise_agent(session_id: str):
    # Close the setup connection - agent uses its own connections via run_sql
    if "connection" in sessions[session_id]:
        sessions[session_id]["connection"].close()
    llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=sessions[session_id]["api_key"])
    agent = create_agent(model=llm, tools=[run_sql])
    sessions[session_id]["agent"] = agent
    return {"status": "ok"}

@app.post("/analysis-outer-loop")
def analysis_outer_loop(session_id: str, user_input: str):
    # Load the Data schema understanding from the setup conversation into the system prompt
    with open(f"sessions/{session_id}/setup_conversation.json") as f:
        schema_context = json.load(f)
        schema_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in schema_context])

    # Loading information from Chroma that matches the user input (excluding setup conversation, redundant)
    results = sessions[session_id]["chroma_collection"].query(
        query_texts=[user_input],
        n_results=3,
        where={"type": {"$ne": "setup"}}
    )
    context_past_conversations = "\n\n".join(results['documents'][0])
    # System prompt
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
    5. Present the results in plain language that directly answers the original business question. Be precise — do not confuse attributes of a record (e.g. the city a seller is located in) with the entity itself (e.g. the seller).

    Always ground your answers in actual query results — never guess or estimate.
    Always ask before assuming criteria, thresholds, or definitions that are not explicitly stated in the question.

    When you have completed an analysis with actual query results, end your response with [ANALYSIS COMPLETE].
    Do not include this marker when asking clarifying questions.
    """
    sessions[session_id]["system_prompt"] = system_prompt
    # Calling the agent with user input and sytstem promt incl. setup info and past conversations
    sessions[session_id]["messages"] = []
    response = sessions[session_id]["agent"].invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    })
    # Building messages to summarize and save in Chroma (RAG) later
    sessions[session_id]["messages"].append({
            "role": "user",
            "content": user_input
            })
    sessions[session_id]["messages"].append(
            {
            "role": "assistant",
            "content": response["messages"][-1].content
            })
    satisfied = False
    if "[ANALYSIS COMPLETE]" in response["messages"][-1].content:
        satisfied = True
    return {"response": response["messages"][-1].content, "satisfied": satisfied}

@app.post("/analysis-inner-loop")
def analysis_inner_loop(session_id: str, user_input: str):
    # Adding new user input to messages to summarize and save in Chroma (RAG) later
    sessions[session_id]["messages"].append({"role": "user", "content": user_input})
    # Passing user input, system prompt, and messages as chat history to the agent
    response = sessions[session_id]["agent"].invoke({
        "messages": [{"role": "system", "content":  sessions[session_id]["system_prompt"]}] + sessions[session_id]["messages"]
    })
    # Adding response to to messages summarize and save in Chroma (RAG) later
    sessions[session_id]["messages"].append({"role": "assistant", "content": response["messages"][-1].content})
    satisfied = "[ANALYSIS COMPLETE]" in response["messages"][-1].content
    return {"response": response["messages"][-1].content, "satisfied": satisfied}

# When exiting the "inner loop" add a user message, since last object must be user message to pass to llm
@app.post("/add-user-msg-to-inner-loop")
def add_user_msg_to_inner_loop (session_id: str):
    sessions[session_id]["messages"].append({
        "role": "user", 
        "content": "Please summarise the above analysis concisely for future reference."
    })
    return {"status": "ok"}

@app.post("/analysis-summary")
def analysis_summary (session_id: str):
    # Summarization of messages and saving to Chroma (RAG)
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
    summary = get_claude_client(api_key=sessions[session_id]["api_key"]).messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=summarisation_prompt,
        messages=sessions[session_id]["messages"]
    )

    summary_text = summary.content[0].text
    sessions[session_id]["chroma_collection"].upsert(
        documents=[summary_text],
        ids=[f'analysis_{sessions[session_id]["analysis_count"]}'],
        metadatas=[{"type": "analysis"}]
        )
    return {"status": "ok"}