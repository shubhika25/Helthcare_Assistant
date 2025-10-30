from fastapi import APIRouter, UploadFile, File
from typing import List
from backend.modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from backend.logger import logger
from backend.utils.report_store import save_report_metadata  # ✅ new import

router = APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info("📄 Received uploaded files")

        # ✅ Step 1: Read files in memory
        file_objects = []
        for file in files:
            content = await file.read()
            file_objects.append({
                "filename": file.filename,
                "content": content
            })

        # ✅ Step 2: Pass to vectorstore loader (returns number of chunks per file)
        processed_files = load_vectorstore(file_objects)  # <-- we’ll modify this to return metadata

        # ✅ Step 3: Log metadata locally for tracking
        for record in processed_files:
            save_report_metadata(record["filename"], record["chunks"])

        logger.info("✅ Files processed, embedded, and metadata stored")
        return {"message": f"Processed {len(processed_files)} files successfully."}

    except Exception as e:
        logger.exception("❌ Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
