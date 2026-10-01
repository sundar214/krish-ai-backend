from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client=genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response=client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain the concept of power bi"
)
print(response.text)
for m in client.models.list():
    print(m.name, "-", m.display_name)