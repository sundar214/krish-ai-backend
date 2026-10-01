import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# ========================================
# LOAD ENVIRONMENT
# ========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# ========================================
# CREATE FASTAPI APP
# ========================================

app = FastAPI(
    title="Krish AI API",
    description="RAG backend for Krish AI",
    version="1.0.0"
)


# ========================================
# CORS
# ========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://lighthearted-paletas-0224f1.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# REQUEST MODEL
# ========================================

class QuestionRequest(BaseModel):
    question: str
    session_id: str


# ========================================
# CONVERSATION MEMORY
# ========================================

conversation_history = {}


# ========================================
# GEMINI EMBEDDINGS
# ========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key,
    output_dimensionality=768
)


# ========================================
# CONNECT TO CHROMADB
# ========================================

vectorstore = Chroma(
    collection_name="krish_ai_knowledge",
    persist_directory="./vectorstore",
    embedding_function=embeddings
)


# ========================================
# CREATE RETRIEVER
# ========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ========================================
# GEMINI LLM
# ========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key
)


# ========================================
# FORMAT DOCUMENTS
# ========================================

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


# ========================================
# FORMAT CONVERSATION HISTORY
# ========================================

def format_history(history):

    if not history:
        return "No previous conversation."

    history_parts = []

    for message in history:

        history_parts.append(
            f'{message["role"].upper()}: {message["content"]}'
        )

    return "\n\n".join(history_parts)


# ========================================
# CHECK CASUAL / GREETING MESSAGE
# ========================================

def is_casual_message(question):

    casual_messages = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye"
    }

    return question.lower().strip() in casual_messages


# ========================================
# QUESTION REWRITING PROMPT
# ========================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
Rewrite the current user question into a standalone question.

Use the previous conversation only when necessary to understand
references such as:

"it", "its", "this", "that", "they", "the above", etc.

Rules:

- If the current question is already standalone, return it unchanged.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- Return only the rewritten question.
- Keep the meaning of the user's question unchanged.

-------------------------
PREVIOUS CONVERSATION
-------------------------

{history}

-------------------------
CURRENT QUESTION
-------------------------

{question}

-------------------------
STANDALONE QUESTION
-------------------------
"""
)


# ========================================
# CREATE ANSWER PROMPT
# ========================================

prompt = ChatPromptTemplate.from_template(
    """
You are Krish AI, a helpful AI tutor.

Answer the user's current question using the provided knowledge context
and the relevant previous conversation.

IMPORTANT:

- Focus primarily on the CURRENT QUESTION.
- Use previous conversation only to understand references such as:
  "it", "this", "that", "its", "the above", or follow-up questions.
- Do not repeat the entire previous conversation.
- Do not answer old questions again unless the current question requires it.
- Use the provided knowledge context as the factual knowledge source.
- Do not invent information.
- If the required knowledge is not available in the context, say:

"I don't know based on the provided knowledge."

- Start directly with the answer.
- Do not introduce yourself.
- Do not say "Hello" or "Hi" unnecessarily.
- Give a clear and beginner-friendly explanation.
- Keep the answer reasonably concise.
- Use Markdown formatting when useful.

-------------------------
PREVIOUS CONVERSATION
-------------------------

{history}

-------------------------
KNOWLEDGE CONTEXT
-------------------------

{context}

-------------------------
CURRENT QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""
)


# ========================================
# HOME ENDPOINT
# ========================================

@app.get("/")
def home():

    return {
        "message": "Krish AI Backend is running!"
    }


# ========================================
# ASK ENDPOINT
# ========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    # ------------------------------------
    # Get question and session
    # ------------------------------------

    question = request.question.strip()
    session_id = request.session_id

    print("Session:", session_id)
    print("Question:", question)


    # ------------------------------------
    # Check empty question
    # ------------------------------------

    if not question:

        return {
            "answer": "Please provide a question.",
            "sources": []
        }


    # ------------------------------------
    # Create memory for new session
    # ------------------------------------

    if session_id not in conversation_history:

        conversation_history[session_id] = []


    # ------------------------------------
    # Get previous conversation
    # ------------------------------------

    history = conversation_history[session_id]


    # ------------------------------------
    # Handle casual messages
    # ------------------------------------

    if is_casual_message(question):

        answer = "Hi! What would you like to learn today?"

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        return {
            "question": question,
            "answer": answer,
            "sources": []
        }


    # ------------------------------------
    # Format previous conversation
    # ------------------------------------

    formatted_history = format_history(history)


    # ------------------------------------
    # Rewrite question for retrieval
    # ------------------------------------

    if history:

        rewrite_response = llm.invoke(
            rewrite_prompt.format(
                history=formatted_history,
                question=question
            )
        )

        search_question = rewrite_response.text.strip()

    else:

        search_question = question


    print("Search question:", search_question)


    # ------------------------------------
    # Retrieve knowledge
    # ------------------------------------

    documents = retriever.invoke(search_question)


    # ------------------------------------
    # No knowledge found
    # ------------------------------------

    if not documents:

        answer = "I don't know based on the provided knowledge."

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        return {
            "question": question,
            "answer": answer,
            "sources": []
        }


    # ------------------------------------
    # Build knowledge context
    # ------------------------------------

    context = format_documents(documents)


    # ------------------------------------
    # Generate answer
    # ------------------------------------

    response = llm.invoke(
        prompt.format(
            history=formatted_history,
            context=context,
            question=question
        )
    )

    answer = response.text


    # ------------------------------------
    # Save conversation
    # ------------------------------------

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })


    # ------------------------------------
    # Collect unique sources
    # ------------------------------------

    sources = []

    seen_sources = set()

    for document in documents:

        source = document.metadata.get("source")
        page = document.metadata.get("page")

        source_key = (source, page)

        if source_key not in seen_sources:

            sources.append(
                {
                    "source": source,
                    "page": page
                }
            )

            seen_sources.add(source_key)


    # ------------------------------------
    # Return response
    # ------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }