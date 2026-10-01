import os

from dotenv import load_dotenv

from google import genai
from google.genai import types


# ==============================
# LOAD API KEY
# ==============================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


# ==============================
# GEMINI CLIENT
# ==============================

client = genai.Client(
    api_key=api_key
)


# ==============================
# DOCUMENT EMBEDDING
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
# QUERY EMBEDDING
# ==============================

def create_query_embedding(question):

    formatted_question = (
        f"task: question answering | query: {question}"
    )

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=formatted_question,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values