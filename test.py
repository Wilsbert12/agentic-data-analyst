import chromadb

chroma_client = chromadb.PersistentClient(path="./sessions/<your-session-id>/chroma_db")
collection = chroma_client.get_or_create_collection(name="schema_context")
results = collection.get()
print(f"Documents in Chroma: {len(results['ids'])}")
for id in results['ids']:
    idx = results['ids'].index(id)
    print(f"\n--- {id} ---")
    print(results['documents'][idx])