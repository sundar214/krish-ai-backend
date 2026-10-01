import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)


text = "Pandas is a Python library used for data analysis."

vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 10 values:")
print(vector[:10])