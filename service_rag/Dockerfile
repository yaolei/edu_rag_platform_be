FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认用 uvicorn，端口 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]