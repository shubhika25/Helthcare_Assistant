from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from backend.modules.llm import get_llm_chain
from backend.modules.query_handlers import query_chain
from backend.modules.hybrid_retriver import HybridRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import List, Optional
from backend.logger import logger
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"🧠 User query: {question}")

        # Step 1: Load embeddings
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalassistantindex")

        # Step 2: Hybrid retrieval (PubMed + PDFs + Web)
        retriever_manager = HybridRetriever(index_name=PINECONE_INDEX_NAME, embedder=embed_model)
        docs = retriever_manager.retrieve(question)

        if not docs:
            return {"response": "No relevant information found in trusted sources or PDFs."}

        # Step 3: Wrap docs into retriever
        class CombinedRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = CombinedRetriever(docs)

        # Step 4: Build LLM chain
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("✅ Query processed successfully")
        return result

    except Exception as e:
        logger.exception("❌ Error processing user query")
        return JSONResponse(status_code=500, content={"error": str(e)})
