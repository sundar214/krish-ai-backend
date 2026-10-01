import pymupdf
import chromadb

from embeddings import create_document_embedding


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
# CREATE DOCUMENT EMBEDDING
# ==============================

def create_document_embedding(text, title=None):

    if title is None:
        title = "none"

    formatted_text = (
        f"title: {title} | text: {text}"
    )

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=formatted_text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values


# ==============================
# EXTRACT PDF TEXT
# ==============================

def extract_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        if text.strip():

            pages.append({
                "text": text,
                "page": page_number + 1
            })

    pdf.close()

    return pages


# ==============================
# CREATE CHUNKS
# ==============================

def create_chunks(
    text,
    chunk_size=50,
    overlap=10
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==============================
# BUILD KNOWLEDGE BASE
# ==============================

def build_knowledge_base(pdf_path):

    pages = extract_pdf(pdf_path)

    chunk_number = 0

    for page in pages:

        chunks = create_chunks(
            page["text"]
        )

        for chunk in chunks:

            print(
                f"Creating embedding for chunk "
                f"{chunk_number}..."
            )

            embedding = create_document_embedding(
                chunk,
                title=pdf_path
            )

            chunk_id = f"chunk_{chunk_number}"

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": pdf_path,
                        "page": page["page"]
                    }
                ]
            )

            print(
                f"Stored {chunk_id} "
                f"(Page {page['page']})"
            )

            chunk_number += 1

    print(
        "\nKnowledge base created successfully!"
    )


# ==============================
# MAIN PROGRAM
# ==============================

pdf_path = "pandas.pdf"

print("\n================================")
print("       KNOWLEDGE INGESTION")
print("================================\n")


if collection.count() == 0:

    print("Knowledge base is empty.")
    print("Building knowledge base...\n")

    build_knowledge_base(pdf_path)

else:

    print(
        f"Knowledge base already exists "
        f"({collection.count()} chunks)."
    )

    print(
        "Skipping PDF processing."
    )