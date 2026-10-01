import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)


print("Testing Gemini Embedding 2...")


result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="What is pandas?",
    config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)


print("Embedding created successfully!")

print(
    "Embedding dimensions:",
    len(result.embeddings[0].values)
)