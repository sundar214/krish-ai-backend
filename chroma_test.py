import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Load .env
load_dotenv()


# Gemini Embedding 2
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)


# Check embedding dimension
test_vector = embeddings.embed_query("What is Pandas?")

print("Embedding dimension:", len(test_vector))


# Connect to existing ChromaDB
vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)

print("ChromaDB connected successfully!")


# Test document
documents = [
    Document(
        page_content="Pandas is an open-source Python library used for data analysis.",
        metadata={
            "source": "test",
            "page": 1
        }
    )
]


# Add document
vectorstore.add_documents(documents)

print("Document added successfully!")