from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import list_reports
from backend.middlewares.exception_handlers import catch_exception_middleware
from backend.routes.upload_pdfs import router as upload_router
from backend.routes.ask_questions import router as ask_router
from backend.routes import upload_and_analyze_report

load_dotenv()

app = FastAPI(title="HealthCare Assistant API", version="1.0.0")


# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # or specify list if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Middleware
app.middleware("http")(catch_exception_middleware)

# ✅ Routers
app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(list_reports.router)
app.include_router(upload_and_analyze_report.router)

# ✅ Serve frontend (index.html)
# This makes http://127.0.0.1:8000 open your frontend automatically
#app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

# app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")