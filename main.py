import sqlite3
from pathlib import Path
import pandas as pd
import base64
import os
from dotenv import load_dotenv
import anthropic
import json
import chromadb

RUN_SETUP = False
if RUN_SETUP:
    #Setting up the Claude client
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Creating and connecting an in-memory database
    connection = sqlite3.connect(":memory:")

    # Load each CSV into pandas then into SQLite
    for file in Path("data/").glob("*.csv"):
        df = pd.read_csv(file)
        df.to_sql(file.stem, connection, if_exists="replace", index=False)

    # Committing the changes
    connection.commit()

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

    # Function to return basic analytics as a string
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

    # Save full conversation to file for reference (purely for convenience)
    with open("sessions/setup_conversation.json", "w") as f:
        json.dump(messages[1:], f, indent=2)
    print("Setup conversation saved to sessions/setup_conversation.json")

    # Store confirmed schema understanding in Chroma
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="schema_context")
    collection.upsert(
        documents=[conversation_text],
        ids=["setup_conversation"]
    )
    print("Schema understanding stored in Chroma.")