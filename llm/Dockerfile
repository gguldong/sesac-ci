# # CUDA 12.4 기반 devel 이미지 (Ubuntu 22.04) 사용 → GPU 지원 필요 시 사용
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# # 대화형 프롬프트 비활성화 설정
# ENV DEBIAN_FRONTEND=noninteractive

# # 필수 패키지 설치: Python 3.10, venv, 개발 헤더, 빌드 도구 등
# RUN apt-get update && apt-get install -y \
#    python3.10 \
#    python3.10-venv \
#    python3.10-dev \
#    cmake \
#    gcc \
#    g++ \
#    git \
#    curl \
#    && rm -rf /var/lib/apt/lists/*

# # pip 설치
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# # Python 가상환경 생성 및 활성화
# RUN python3.10 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # CUDA 빌드를 위한 환경 변수 설정 (GPU 지원 활성화)
# ENV PATH=/usr/local/cuda-12.4/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# # 작업 디렉토리 설정
# WORKDIR /app

# # requirements.txt 복사 및 의존성 설치
# COPY requirements.txt ./ 
# RUN pip install --upgrade pip && \
#    pip install --no-cache-dir -r requirements.txt

# # llama-cpp-python 설치 (CUDA 빌드 비활성화하여 컴파일 오류 방지)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install --no-cache-dir llama-cpp-python --prefer-binary

# # faiss-gpu를 CUDA 지원 버전으로 강제 재설치 (Python 3.10용)
# RUN pip install --force-reinstall --no-cache-dir faiss-gpu==1.7.2 --extra-index-url https://pypi.nvidia.com

# # 소스 코드 전체 복사
# COPY . .

# # 실행 스크립트에 실행 권한 부여
# RUN chmod +x /app/run_llm_service.sh /app/rag/run_api_server.sh

# # 컨테이너 실행 시, 두 스크립트를 백그라운드에서 동시에 실행하고 대기
# CMD ["/bin/sh", "-c", "/app/run_llm_service.sh & /app/rag/run_api_server.sh && wait"]

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 대화형 프롬프트 비활성화 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치: Python 3.10, venv, 개발 헤더, 빌드 도구 등
RUN apt-get update && apt-get install -y \
   python3.10 \
   python3.10-venv \
   python3.10-dev \
   cmake \
   gcc \
   g++ \
   git \
   curl \
   && rm -rf /var/lib/apt/lists/*

# pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Python 가상환경 생성 및 활성화
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CUDA 빌드를 위한 환경 변수 설정 (GPU 지원 활성화)
ENV PATH=/usr/local/cuda-12.4/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
RUN pip install --upgrade pip && \
   pip install --no-cache-dir \
   fastapi==0.115.11 \
   uvicorn==0.34.0 \
   pydantic==2.10.6 \
   pydantic-settings==2.8.1 \
   numpy==1.24.3 \
   SQLAlchemy==2.0.39 \
   python-dotenv==1.0.1

# HTTP 및 통신 관련 패키지
RUN pip install --no-cache-dir \
   httpx==0.28.1 \
   httpx-sse==0.4.0 \
   aiohttp==3.11.14 \
   requests==2.32.3 \
   requests-toolbelt==1.0.0

# AI/LLM 관련 패키지
RUN pip install --no-cache-dir \
   langchain==0.3.21 \
   langchain-core==0.3.46 \
   langchain-community==0.3.20 \
   langchain-huggingface==0.1.2 \
   langchain-cohere==0.4.3 \
   langchain-openai==0.3.9 \
   langchain-text-splitters==0.3.7 \
   tiktoken==0.9.0 \
   sentence-transformers==3.4.1 \
   transformers==4.49.0 \
   openai==1.67.0 \
   cohere==5.14.0

# CUDA 및 ML 관련 패키지
RUN pip install --no-cache-dir \
   torch==2.6.0 \
   faiss-gpu==1.7.2 \
   scikit-learn==1.6.1 \
   scipy==1.15.2

# llama-cpp-python 설치 (CUDA 활성화)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON" pip install --no-cache-dir llama-cpp-python==0.3.8 --prefer-binary

# 데이터베이스 관련
RUN pip install --no-cache-dir \
   mysql-connector-python==9.2.0

# 유틸리티 패키지
RUN pip install --no-cache-dir \
   PyYAML==6.0.2 \
   tqdm==4.67.1 \
   pillow==11.1.0 \
   orjson==3.10.15 \
   dataclasses-json==0.6.7

# 소스 코드 전체 복사
COPY . .

# 실행 스크립트에 실행 권한 부여
RUN chmod +x /app/run_llm_service.sh /app/rag/run_api_server.sh

# 포트 노출
EXPOSE 8001
#
# 컨테이너 실행 시, 두 스크립트를 백그라운드에서 동시에 실행하고 대기
CMD ["/bin/sh", "-c", "/app/run_llm_service.sh & /app/rag/run_api_server.sh && wait"]