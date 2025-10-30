from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.utils.report_store import list_reports

router = APIRouter()

@router.get("/list_reports/")
async def get_uploaded_reports():
    try:
        reports = list_reports()
        if not reports:
            return {"message": "No reports uploaded yet."}
        return {"uploaded_reports": reports}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
