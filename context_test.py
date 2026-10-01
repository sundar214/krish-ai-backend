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

documents = retriever.invoke(question)


context_parts = []

for i, document in enumerate(documents):

    context_parts.append(
        f"""
SOURCE {i + 1}
Source: {document.metadata.get("source")}
Page: {document.metadata.get("page")}

{document.page_content}
"""
    )


context = "\n".join(context_parts)


print("\n==============================")
print("CONTEXT FOR GEMINI")
print("==============================")

print(context)