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


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


question = "What is Pandas?"


results = retriever.invoke(question)


print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================\n")


for i, document in enumerate(results):

    print(f"DOCUMENT {i + 1}")

    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

    print("\n------------------------------")