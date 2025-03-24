# Python 3.12 slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# MariaDB 설치
RUN apt-get update && apt-get install -y \
    # mariadb-server \
    # mariadb-client \
    git \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# MySQL 설정 파일 복사 (옵션)
# COPY my.cnf /etc/mysql/my.cnf

# 데이터베이스 초기화 스크립트 복사
# COPY init.sql /docker-entrypoint-initdb.d/

# 필수 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 


# 소스 코드 복사
COPY . .

# MariaDB 서비스 시작
# RUN service mariadb start

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

