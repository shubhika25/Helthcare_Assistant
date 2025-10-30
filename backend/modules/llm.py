import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ✅ Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm_chain(retriever):
    # ---------------------------
    # 🧠 Initialize Groq LLM
    # ---------------------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # ---------------------------
    # 🧩 Create Prompt
    # ---------------------------
    prompt = ChatPromptTemplate.from_template("""
You are  an AI-powered assistant trained to help users understand medical documents and health-related questions.

Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.

---
🔍 **Context:**
{context}

🙋‍♂️ **User Question:**
{question}

---
💬 **Answer:**
- Respond in a calm, factual, and respectful tone.
- Use simple explanations when needed.
- If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts or provide information not present in the context.
- Keep your answers concise and to the point.
- Act like a professional medical assistant.                                                                                        
""")

    # ---------------------------
    # ⚙️ LCEL Chain (new version)
    # ---------------------------
    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
