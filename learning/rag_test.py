import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. CREATE CHROMADB CLIENT
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./vectorstore"
)


# ============================================================
# 4. CREATE / GET COLLECTION
# ============================================================

collection = chroma_client.get_or_create_collection(
    name="krish_ai_knowledge"
)


# ============================================================
# 5. OUR KNOWLEDGE
# ============================================================

documents = [

    "Python is a high-level programming language used for "
    "web development, data analysis, automation, and artificial intelligence.",

    "A Python list is an ordered and mutable collection "
    "that can store multiple values.",

    "A Python dictionary stores data in key-value pairs. "
    "Each key is used to access its corresponding value.",

    "Python functions are reusable blocks of code designed "
    "to perform a specific task.",

    "A Python tuple is an ordered collection that is immutable, "
    "meaning its elements cannot be changed after creation."

]


# ============================================================
# 6. CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values


# ============================================================
# 7. STORE KNOWLEDGE IN CHROMADB
# ============================================================

print("Creating embeddings and storing knowledge...\n")


for index, document in enumerate(documents):

    embedding = create_embedding(document)

    collection.upsert(
        ids=[f"doc_{index}"],
        documents=[document],
        embeddings=[embedding]
    )

    print(f"Stored document {index + 1}")


print("\nKnowledge stored successfully!")


# ============================================================
# 8. ASK A QUESTION
# ============================================================

question = "What is a Python dictionary?"


print("\nQuestion:")
print(question)


# ============================================================
# 9. CREATE EMBEDDING FOR THE QUESTION
# ============================================================

question_embedding = create_embedding(question)


# ============================================================
# 10. SEARCH CHROMADB
# ============================================================

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# ============================================================
# 11. GET RELEVANT DOCUMENTS
# ============================================================

retrieved_documents = results["documents"][0]


print("\nRelevant knowledge:")

for document in retrieved_documents:

    print("-", document)


# ============================================================
# 12. COMBINE RETRIEVED DOCUMENTS
# ============================================================

context = "\n\n".join(retrieved_documents)


# ============================================================
# 13. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are Krish AI, a helpful AI tutor.

Answer the user's question using only the provided knowledge.

If the answer is not available in the provided knowledge,
say that you don't know based on the provided knowledge.

Keep the explanation simple and clear.

Knowledge:
{context}

Question:
{question}

Answer:
"""


    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# 14. GENERATE FINAL KRISH AI ANSWER
# ============================================================

answer = generate_answer(
    question,
    context
)


# ============================================================
# 15. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("              KRISH AI")
print("========================================")

print("\nAnswer:")
print(answer)

print("\n========================================")
print("              RAG COMPLETE")
print("========================================")