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
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

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

#Setting up the Claude client
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

messages = []
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
    print("Please enter a question.")
    user_input = "\n".join(lines)
messages.append(
        {
        "role": "user",
        "content": user_input
        })

#Load the Data schema understanding from the setup conversation into the system prompt
with open("sessions/setup_conversation.json") as f:
    schema_context = json.load(f)
    schema_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in schema_context])

# Loading information from Chroma that matches the user input (excluding setup conversation, redundant)
results = collection.query(
    query_texts=[user_input],
    n_results=3,
    where={"type": {"$ne": "setup"}}
)
context_past_conversations = "\n\n".join(results['documents'][0])

system_prompt = f"""
You are an expert data analyst. You have full knowledge of the dataset schema and confirmed handling rules from the setup phase:

{schema_text}

Additional context from past analyses, if available:
{context_past_conversations}

When the user asks a business question, follow these steps:

1. Interpret the question and identify what data is needed to answer it.
2. If the question requires multiple queries, break it down into clear steps.
3. If anything is unclear, ask the user clarifying questions before proceeding.
4. If there are multiple steps, walk the user through your planned approach and confirm before executing.
5. Execute the necessary queries.
6. Present the results in plain language that directly answers the original business question.

Always ground your answers in actual query results — never guess or estimate.
"""


print("Analysing business problem...")


#setting up the agent
connection.close()
db = SQLDatabase.from_uri("sqlite:///sessions/database.sqlite")

llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY"))

agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True
)
#Setting up counter outside conversation loop to set documnent IDs for Chroma (RAG)
analysis_count = 0

response = agent.invoke({
    "input": user_input,
    "system": system_prompt
})

messages = []
messages.append({
        "role": "user",
        "content": user_input
        })
messages.append(
        {
        "role": "assistant",
        "content": response["output"]
        })
messages.append({
    "role": "user", 
    "content": "Please summarise the above analysis concisely for future reference."
    })
summarisation_prompt = "summarize the following conversation for easy retrival of relevant information"
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

print(response["output"])