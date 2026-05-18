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
# 2. DATABASE CONNECTION
# =============================================================================

# Creating and connecting an SQL database
connection = sqlite3.connect("sessions/database.sqlite")
# Create and connect to Chroma database (RAG)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="schema_context")

# =============================================================================
# 3. SETUP PHASE — Schema Understanding
# =============================================================================

# Setting up the Claude client
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Starting a new session: Option to delete content from previous sessions
print("Start fresh or continue previous session?")
print("NEW — deletes all previous analyses from memory and starts fresh")
print("CONTINUE — loads previous analyses and continues from where you left off")
session_choice = input().strip().upper()
if session_choice == "NEW":
    existing = collection.get()
    ids_to_delete = [id for id in existing['ids'] if id.startswith('analysis_')]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Cleared {len(ids_to_delete)} previous analyses.")

RUN_SETUP = False
if RUN_SETUP:
    # Load each CSV into pandas then into SQLite
    connection = sqlite3.connect("sessions/database.sqlite")
    for file in Path("data/").glob("*.csv"):
        df = pd.read_csv(file)
        df.to_sql(file.stem, connection, if_exists="replace", index=False)

    # Build context message for Claude API from files in the context/ folder
    # Supports: .txt (data cards), .csv (data dictionaries), .jpg/.png (schema images via vision)
    messages=[{"role": "user"}]
    content=[] 
    for file in Path("context/").iterdir():
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
                        "media_type": "image/png",
                        "data": file_content
                    }
                }
                )
        else: pass

    # Function to return basic analytics as a string to pass to llm
    def basic_analysis (df):
        return (str(df.shape) + "\n\n" + 
                df.dtypes.to_string() + "\n\n" + 
                df.isnull().mean().to_string() + "\n\n" + 
                df.nunique().to_string() + "\n\n" + 
                df.describe().to_string())

    # Adding 5 sample rows with explanatory intro for context
    for file in Path("data/").glob("*.csv"):
        content.append(
            {
            "type": "text",
            "text": f"Sample rows from table '{file.stem}' (5 rows, all columns):"
            })
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
    print("Analysing data...")
    response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system_prompt,
            messages=messages)
    print(response.content[0].text)
    messages.append({
        "role": "assistant",
        "content": response.content[0].text
    })

    # Initializing conversational until Claude understands data
    while "[SCHEMA CONFIRMED]" not in response.content[0].text:
        # Accept multi line input
        print("Enter your response (type END on a new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
            # Exit loop if user inputs "EXIT"
            if line.strip() == "EXIT":
                exit()
        user_input = "\n".join(lines)
        # Do not react to empty input
        if not user_input.strip():
            continue
        messages.append(
                {
                "role": "user",
                "content": user_input
                })
        
        # Loop back to Claude including last response and new user input
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system_prompt,
            messages=messages
        )
        messages.append(
                {
                "role": "assistant",
                "content": response.content[0].text
                })
        print(response.content[0].text)

    # After all questions are resolved and conversation is ended
    # Save the conversation except initial user input (text, images etc.) into a string
    conversation_text = ""
    for message in messages[1:]:
        role = message["role"]
        content = message["content"]
        conversation_text += f"{role}: {content}\n\n"

    # Save full conversation to file for reference. Used to bypass Chroma (to always give llm setup context)
    with open("sessions/setup_conversation.json", "w") as f:
        json.dump(messages[1:], f, indent=2)
    print("Setup conversation saved to sessions/setup_conversation.json")

    # Store confirmed schema understanding in Chroma
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="schema_context")
    collection.upsert(
        documents=[conversation_text],
        ids=["setup_conversation"],
        metadatas=[{"type": "setup"}]
    )
    print("Schema understanding stored in Chroma.")

tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(tables)

# =============================================================================
# 4. ANALYTICAL PHASE — Core Loop
# =============================================================================

# Setup once before the core loop
# Setting up the agent
connection.close()
llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY"))
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
# Defining the model
agent = create_agent(
    model=llm,
    tools=[run_sql]
)

# Setting up counter outside conversation loop to set documnent IDs for Chroma (RAG)
analysis_count = 0

# Load the Data schema understanding from the setup conversation into the system prompt
with open("sessions/setup_conversation.json") as f:
    schema_context = json.load(f)
    schema_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in schema_context])

# Core loop
while True:
    # Accept multi line input
    print("What is the business question you want to analyse? (type END on a new line when done):")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
        # Exit loop if user inputs "EXIT"
        if line.strip() == "EXIT":
            exit()
    user_input = "\n".join(lines)
    # Do not react to empty input
    if not user_input.strip():
        continue
    # Loading information from Chroma that matches the user input (excluding setup conversation, redundant)
    results = collection.query(
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
    # Calling the agent with user input and sytstem promt incl. setup info and past conversations
    response = agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    })
    # Building messages to summarize and save in Chroma (RAG) later
    messages = []
    messages.append({
            "role": "user",
            "content": user_input
            })
    messages.append(
            {
            "role": "assistant",
            "content": response["messages"][-1].content
            })
    # Starting an "inner loop" after each initial question until answer is satisfactory
    satisfied = False
    # Asking FUP question and accepting multiline input
    while not satisfied:
        # Printing the response
        print(response["messages"][-1].content)
        if "[ANALYSIS COMPLETE]" in response["messages"][-1].content:
            print("\nDoes this answer your question? Type YES or provide follow-up (END to submit):")
        else:
            # just capture follow-up input without the satisfaction prompt
            print("\nProvide follow-up (type END when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            if "[ANALYSIS COMPLETE]" in response["messages"][-1].content and line.strip() == "YES":
                satisfied = True
                break
            if line.strip() == "EXIT":
                exit()
            lines.append(line)
        if satisfied:
            break
        inner_user_input = "\n".join(lines)
        if not inner_user_input.strip():
            continue
        # Adding new user input to messages to summarize and save in Chroma (RAG) later
        messages.append({"role": "user", "content": inner_user_input})
        # Passing user input, system prompt, and messages as chat history to the agent
        response = agent.invoke({
            "messages": [{"role": "system", "content": system_prompt}] + messages
        })
        # Adding response to to messages summarize and save in Chroma (RAG) later
        messages.append({"role": "assistant", "content": response["messages"][-1].content})
    # When exiting the "inner loop" add a user message, since last object must be user message to pass to llm
    messages.append({
        "role": "user", 
        "content": "Please summarise the above analysis concisely for future reference."
        })
    
    if satisfied:
        # Summarization of messages and saving to Chroma (RAG)
        summarisation_prompt = summarisation_prompt = """
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
        summary = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=summarisation_prompt,
            messages=messages
        )

        summary_text = summary.content[0].text
        analysis_count += 1
        collection.upsert(
            documents=[summary_text],
            ids=[f"analysis_{analysis_count}"],
            metadatas=[{"type": "analysis"}]
            )