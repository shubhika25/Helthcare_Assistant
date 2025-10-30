FROM python:3.11-slim

RUN pip install --upgrade pip && pip install uv

WORKDIR /app

COPY backend/requirements.txt ./

RUN uv pip install --system --no-cache -r requirements.txt

COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
