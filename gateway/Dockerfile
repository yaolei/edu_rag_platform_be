FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

# 默认用 uvicorn，端口 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]