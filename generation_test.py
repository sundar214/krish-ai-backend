import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma


load_dotenv()


# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)


# ChromaDB
vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# User question
question = "What is Pandas?"


# Retrieve relevant documents
documents = retriever.invoke(question)


# Build context
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


# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# Prompt
prompt = f"""
You are Krish AI, a helpful AI tutor.

Answer the question using only the provided context.

If the answer is not available in the context, say:

"I don't know based on the provided knowledge."

Explain the answer in a simple and beginner-friendly way.

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""


# Generate answer
response = llm.invoke(prompt)


print("\n==============================")
print("KRISH AI")
print("==============================\n")

print(response.text)