FROM python:3.11-slim

WORKDIR /app

COPY wheels/ ./wheels/
RUN pip install \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
    --trusted-host mirrors.aliyun.com \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --trusted-host mirrors.bfsu.edu.cn \
    --find-links ./wheels \
    torch==2.9.1+cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


COPY . .

# 默认用 uvicorn，端口 8000
CMD ["uvicorn", "service_rag.app.main:app", "--host", "0.0.0.0", "--port", "8001"]