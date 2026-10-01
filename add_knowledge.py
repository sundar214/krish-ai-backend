import os
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# -----------------------------
# Check PDF argument
# -----------------------------

if len(sys.argv) < 2:
    print("Please provide a PDF file.")
    print("Example:")
    print("python add_knowledge.py powerbi.pdf")
    sys.exit()


pdf_path = sys.argv[1]

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit()


# -----------------------------
# Load PDF
# -----------------------------

print(f"\nLoading: {pdf_path}")

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# -----------------------------
# Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# -----------------------------
# Add metadata
# -----------------------------

for chunk in chunks:
    chunk.metadata["source"] = os.path.basename(pdf_path)


# -----------------------------
# Gemini Embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key,
    output_dimensionality=768
)


# -----------------------------
# Existing ChromaDB
# -----------------------------

vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


# -----------------------------
# Add chunks safely
# -----------------------------

print("\nAdding knowledge to ChromaDB...")

batch_size = 20

total_chunks = len(chunks)

for start in range(0, total_chunks, batch_size):

    end = min(start + batch_size, total_chunks)

    batch = chunks[start:end]

    print(
        f"Adding chunks {start + 1}-{end} "
        f"of {total_chunks}..."
    )

    vectorstore.add_documents(batch)

    if end < total_chunks:
        print("Waiting before next batch...")
        time.sleep(15)


# -----------------------------
# Finished
# -----------------------------

print("\nKnowledge added successfully!")
print(f"Source: {os.path.basename(pdf_path)}")
print(f"Chunks added: {total_chunks}")