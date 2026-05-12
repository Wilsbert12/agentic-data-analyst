import sqlite3
from pathlib import Path
import pandas as pd
import base64

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
messages[0]["content"] = content
