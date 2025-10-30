# test_hello.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env vars
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

# Initialize model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3-8b-instant"
)

# Create a simple prompt
prompt = ChatPromptTemplate.from_template("""
You are a friendly AI assistant.
Answer politely and concisely.

User: {question}
AI:
""")

# Create a simple chain (no retriever)
chain = prompt | llm | StrOutputParser()

# Test with sample query
query = "hello, how are you?"
response = chain.invoke({"question": query})

print("\n=== 💬 MODEL RESPONSE ===")
print(response)
