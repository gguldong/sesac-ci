# # Python 3.12 slim 이미지 사용
# FROM python:3.12-slim

# # 작업 디렉토리 설정
# WORKDIR /app

# # MariaDB 설치
# RUN apt-get update && apt-get install -y \
#     mariadb-server \
#     mariadb-client \
#     git \
#     build-essential \
#     cmake \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # MySQL 설정 파일 복사 (옵션)
# # COPY my.cnf /etc/mysql/my.cnf

# # 데이터베이스 초기화 스크립트 복사
# # COPY init.sql /docker-entrypoint-initdb.d/

# # 필수 패키지 설치
# COPY requirements.txt ./
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt 


# # 소스 코드 복사
# COPY . .

# # MariaDB 서비스 시작
# # RUN service mariadb start

# # FastAPI 서버 실행
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]







# # Python 3.10 이미지 사용 (requirements.txt의 패키지 호환성을 고려)
# FROM python:3.10-slim

# # 작업 디렉토리 설정
# WORKDIR /app

# # 필요한 시스템 패키지 설치
# RUN apt-get update && apt-get install -y \
#     mariadb-server \
#     mariadb-client \
#     build-essential \
#     cmake \
#     git \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # 의존성 파일 복사 및 설치
# COPY requirements.txt ./
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir numpy==1.26.2 && \
#     pip install --no-cache-dir -r requirements.txt --ignore-installed


# # 애플리케이션 코드 복사
# COPY . .

# # # 환경 변수 설정 (필요시 수정)
# # ENV PYTHONPATH=/app

# # 포트 노출
# EXPOSE 8000

# # FastAPI 서버 실행
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    mariadb-client \
    build-essential \
    cmake \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 필수 패키지만 직접 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install numpy==1.24.4 && \
    pip install fastapi==0.115.11 uvicorn==0.34.0 pydantic==2.10.6 python-dotenv==1.0.1 itsdangerous

# 다른 필수 패키지들을 개별적으로 설치 - 실패해도 계속 진행
RUN pip install langchain==0.3.21 || true
RUN pip install langchain-core==0.3.45 || true
RUN pip install openai==1.66.5 || true
RUN pip install SQLAlchemy==2.0.39 || true
RUN pip install python-multipart==0.0.20 || true
RUN pip install requests==2.32.3 || true
RUN pip install pymysql || true

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]