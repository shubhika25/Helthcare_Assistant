import json
import os
from datetime import datetime

REPORT_LOG_FILE = "uploaded_reports.json"

def save_report_metadata(filename: str, chunk_count: int):
    """Save metadata for uploaded lab reports."""
    data = []
    if os.path.exists(REPORT_LOG_FILE):
        with open(REPORT_LOG_FILE, "r") as f:
            data = json.load(f)

    entry = {
        "filename": filename,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": chunk_count
    }

    data.append(entry)
    with open(REPORT_LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

def list_reports():
    """Retrieve all uploaded lab report metadata."""
    if not os.path.exists(REPORT_LOG_FILE):
        return []
    with open(REPORT_LOG_FILE, "r") as f:
        return json.load(f)
