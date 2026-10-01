import chromadb

chroma_client = chromadb.PersistentClient(path="./vectorstore")

collection = chroma_client.get_or_create_collection(
    name="krish_ai_knowledge"
)

# Find the temporary test document
results = collection.get(
    where={"source": "test"}
)

print("Test documents found:", len(results["ids"]))

if results["ids"]:
    collection.delete(ids=results["ids"])
    print("Temporary test documents deleted.")
else:
    print("No test documents found.")

print("Remaining documents:", collection.count())