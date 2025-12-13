FROM python:3.11-slim

WORKDIR /app

# 换国内源 + BuildKit 缓存
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set global.trusted-host mirrors.aliyun.com

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

# 默认用 uvicorn，端口 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]