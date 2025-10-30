# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import JSONResponse
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_groq import ChatGroq  # ✅ Groq integration
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# import os

# router = APIRouter()
# load_dotenv()

# # ---------------------------
# # 🧠 Initialize Groq LLM
# # ---------------------------
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(
#     groq_api_key=GROQ_API_KEY,
#     model_name="llama-3.3-70b-versatile",
#     temperature=0.2,
# )

# # ---------------------------
# # 📋 Prompt Template
# # ---------------------------
# analysis_prompt = ChatPromptTemplate.from_template("""
# You are a medical analyst AI assistant.
# You are analyzing a patient’s lab report.

# Report Text:
# {report_text}

# Your tasks:
# 1. Extract key medical parameters (e.g., Hemoglobin, Glucose, Cholesterol, etc.)
# 2. For each parameter, give: value, normal range status, and significance.
# 3. Summarize overall health status in 2–3 lines.
# 4. Respond strictly in JSON format.

# Example JSON:
# {
#   "parameters": [
#     {"name": "Hemoglobin", "value": "11.2 g/dL", "status": "Low", "explanation": "Suggests mild anemia"},
#     {"name": "Glucose", "value": "108 mg/dL", "status": "Slightly High", "explanation": "Possible prediabetic range"}
#   ],
#   "summary": "Mild anemia and borderline glucose elevation observed."
# }
# """)

# # ---------------------------
# # 🚀 Route for Lab Report Analysis
# # ---------------------------
# @router.post("/analyze_lab_report/")
# async def analyze_lab_report(file: UploadFile = File(...)):
#     try:
#         # ✅ Save uploaded PDF temporarily
#         os.makedirs("./uploaded_docs", exist_ok=True)
#         file_path = f"./uploaded_docs/{file.filename}"
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         # ✅ Extract text
#         loader = PyPDFLoader(file_path)
#         pages = loader.load()
#         report_text = "\n".join([p.page_content for p in pages])

#         # ✅ Create LCEL chain
#         chain = analysis_prompt | llm | StrOutputParser()

#         # ✅ Run chain
#         result = chain.invoke({"report_text": report_text})

#         return {"filename": file.filename, "analysis": result}

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
