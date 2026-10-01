import os

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# --------------------------------
# LOAD API KEY
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# --------------------------------
# GEMINI EMBEDDINGS
# --------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key,
    output_dimensionality=768
)


# --------------------------------
# CONNECT TO CHROMADB
# --------------------------------

vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


# --------------------------------
# CREATE RETRIEVER
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------
# GEMINI LLM
# --------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)


# --------------------------------
# FORMAT RETRIEVED DOCUMENTS
# --------------------------------

def format_documents(documents):

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

    return "\n".join(context_parts)


# --------------------------------
# CREATE PROMPT
# --------------------------------

prompt = ChatPromptTemplate.from_template("""
You are Krish AI, a helpful AI tutor.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:

"I don't know based on the provided knowledge."

Do not invent information.

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
""")


# --------------------------------
# CREATE RAG CHAIN
# --------------------------------

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# --------------------------------
# ASK QUESTION
# --------------------------------

question = input("\nAsk Krish AI: ")


# --------------------------------
# RETRIEVE DOCUMENTS
# --------------------------------

documents = retriever.invoke(question)


# --------------------------------
# CHECK RESULTS
# --------------------------------

if not documents:

    print("\n================================")
    print("       NO RELEVANT KNOWLEDGE")
    print("================================\n")

    print("I don't know based on the provided knowledge.")

    exit()


# --------------------------------
# SHOW RETRIEVED CONTEXT
# --------------------------------

context = format_documents(documents)

print("\n================================")
print("       RETRIEVED CONTEXT")
print("================================")

print(context)


# --------------------------------
# RUN RAG CHAIN
# --------------------------------

response = rag_chain.invoke(question)


# --------------------------------
# DISPLAY ANSWER
# --------------------------------

print("\n================================")
print("             KRISH AI")
print("================================\n")

print(response.text)


# --------------------------------
# SHOW SOURCES
# --------------------------------

print("\n================================")
print("             SOURCES")
print("================================\n")

for document in documents:

    print(
        f"📄 {document.metadata.get('source')} "
        f"| Page {document.metadata.get('page')}"
    )


# --------------------------------
# COMPLETE
# --------------------------------

print("\n================================")
print("          RAG COMPLETE")
print("================================")