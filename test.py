import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="schema_context")
results = collection.get()
print(f"Documents in Chroma: {len(results['ids'])}")
print(results['ids'])