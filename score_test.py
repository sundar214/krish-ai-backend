import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)


vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


question = "What is Pandas?"


results = vectorstore.similarity_search_with_score(
    question,
    k=5
)


print("\n==============================")
print("SEARCH RESULTS")
print("==============================\n")


for i, (document, score) in enumerate(results):

    print(f"RESULT {i + 1}")

    print("Score:", score)

    print("Content:")
    print(document.page_content)

    print("Metadata:")
    print(document.metadata)

    print("\n------------------------------")