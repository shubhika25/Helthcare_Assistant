# routes/lab_reports.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.modules.load_vectorstore import load_vectorstore
from dotenv import load_dotenv
import os

router = APIRouter()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
)

analysis_prompt = ChatPromptTemplate.from_template("""
You are a medical analyst AI assistant analyzing a patient’s lab report.

Report Text:
{report_text}

Return in **strict JSON format** as below:
{{
  "parameters": ["Parameter name and value if available"],
  "summary": "A short summary of the key findings and abnormalities."
}}
""")

@router.post("/upload_and_analyze_lab_report/")
async def upload_and_analyze_lab_report(file: UploadFile = File(...)):
    import re, json, traceback
    try:
        os.makedirs("./uploaded_docs", exist_ok=True)
        path = f"./uploaded_docs/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        print(f"✅ Uploaded file saved at: {path}")

        loader = PyPDFLoader(path)
        pages = loader.load()
        report_text = "\n".join([p.page_content for p in pages])
        print(f"✅ Extracted {len(report_text)} characters from PDF")

        chain = analysis_prompt | llm | StrOutputParser()
        print("🧠 Invoking Groq LLM...")
        raw_analysis = chain.invoke({"report_text": report_text})
        print(f"✅ LLM Response:\n{raw_analysis}")

        # ✅ Extract clean JSON
        json_match = re.search(r"```json([\s\S]*?)```|```([\s\S]*?)```", raw_analysis)
        if json_match:
            json_text = (json_match.group(1) or json_match.group(2)).strip()
        else:
            # fallback: extract from plain braces
            fallback = re.search(r"\{[\s\S]*\}", raw_analysis)
            json_text = fallback.group(0).strip() if fallback else None

        analysis = {}
        if json_text:
            try:
                analysis = json.loads(json_text)
            except json.JSONDecodeError as e:
                print("⚠️ JSON parse failed:", e)
                analysis = {"raw_text": raw_analysis}
        else:
            analysis = {"raw_text": raw_analysis}

        # ✅ Add to vector DB
        file.file.seek(0)
        load_vectorstore([{"filename": file.filename, "content": file.file.read()}])
        print("📚 Added to vector database")

        return {
            "filename": file.filename,
            "analysis": analysis,  # now this is clean JSON, not a string
            "message": "Report analyzed and stored in vector database."
        }

    except Exception as e:
        print("❌ Exception:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
