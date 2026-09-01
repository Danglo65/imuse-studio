# CPU image, mock ML mode only -- for developing/testing the app itself
# (datasets, jobs, UI) without a GPU. For real training/generation use
# Dockerfile.gpu on a cloud GPU box.
FROM python:3.11-slim

WORKDIR /app/backend

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV IMUSE_MOCK_ML=1
ENV IMUSE_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
