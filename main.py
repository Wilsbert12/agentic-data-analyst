import sqlite3
from pathlib import Path
import pandas as pd

# Creating and connecting an in-memory database
connection = sqlite3.connect(":memory:")

# Load each CSV into pandas then into SQLite
for file in Path("data/").glob("*.csv"):
    df = pd.read_csv(file)
    df.to_sql(file.stem, connection, if_exists="replace", index=False)

# Committing the changes
connection.commit()