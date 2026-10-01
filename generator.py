import os

from dotenv import load_dotenv

from google import genai


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
# GENERATE ANSWER
# ==============================

def generate_answer(question, context):

    prompt = f"""
You are Krish AI, a helpful AI tutor.

Answer the user's question using only
the provided context.

The context contains SOURCE labels,
page numbers, and distances.

Use the provided source information when
explaining where the answer came from.

Do not invent source names or page numbers.

If the answer is not available in the
provided context, say:

"I don't know based on the provided knowledge."

Do not make up information.

Explain the answer in a simple and
beginner-friendly way.

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

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text