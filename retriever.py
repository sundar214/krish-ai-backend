import chromadb

from embeddings import create_query_embedding


# ==============================
# CONNECT TO CHROMADB
# ==============================

chroma_client = chromadb.PersistentClient(
    path="./vectorstore"
)

collection = chroma_client.get_or_create_collection(
    name="krish_ai_knowledge"
)


# ==============================
# RETRIEVE KNOWLEDGE
# ==============================

def retrieve_knowledge(
    question,
    n_results=5,
    max_distance=0.70,
    max_context_results=3
):

    # Create vector for user's question
    question_embedding = create_query_embedding(
        question
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]


    # ==============================
    # RAW RESULTS
    # ==============================

    print("\n================================")
    print("       RAW RETRIEVAL RESULTS")
    print("================================\n")

    for i in range(len(documents)):

        print(f"Result {i + 1}")
        print(f"Distance: {distances[i]}")
        print(f"Text: {documents[i][:200]}")
        print()


    # ==============================
    # FILTER RESULTS
    # ==============================

    filtered_documents = []
    filtered_distances = []
    filtered_metadatas = []

    for i in range(len(documents)):

        if distances[i] <= max_distance:

            filtered_documents.append(
                documents[i]
            )

            filtered_distances.append(
                distances[i]
            )

            filtered_metadatas.append(
                metadatas[i]
            )


    # ==============================
    # FILTERING INFORMATION
    # ==============================

    print("\n================================")
    print("       FILTERING RESULTS")
    print("================================\n")

    print(
        f"Raw results     : {len(documents)}"
    )

    print(
        f"Accepted results: "
        f"{len(filtered_documents)}"
    )

    print(
        f"Maximum distance: "
        f"{max_distance}"
    )


    # ==============================
    # LIMIT CONTEXT
    # ==============================

    filtered_documents = (
        filtered_documents[:max_context_results]
    )

    filtered_distances = (
        filtered_distances[:max_context_results]
    )

    filtered_metadatas = (
        filtered_metadatas[:max_context_results]
    )


    # ==============================
    # RETURN RESULTS
    # ==============================

    return {
        "documents": [filtered_documents],
        "distances": [filtered_distances],
        "metadatas": [filtered_metadatas]
    }