FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    mariadb-client \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 필수 Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.115.11 \
    uvicorn==0.34.0 \
    pydantic==2.10.6 \
    pydantic-settings==2.8.1 \
    sqlalchemy==2.0.39 \
    pymysql==1.1.1 \
    python-dotenv==1.0.1 \
    bcrypt==4.3.0 \
    python-multipart==0.0.20 \
    requests==2.32.3 \
    httpx==0.28.1 \
    openai==1.66.5 \
    itsdangerous==2.2.0 \
    email-validator==2.2.0 \
    numpy==1.24.4

# OpenAI 관련 패키지
RUN pip install --no-cache-dir \
    langchain==0.3.21 \
    langchain-core==0.3.45 \
    langchain-openai==0.3.9 || true

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]