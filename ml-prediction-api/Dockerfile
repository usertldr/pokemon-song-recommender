# ml-prediction-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY data/ data/
COPY models/ models/

EXPOSE 8012
CMD ["python", "-m", "api.main"]
