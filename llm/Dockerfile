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
#    build-essential \
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

# # 필수 패키지 설치
# RUN pip install --upgrade pip && \
#    pip install --no-cache-dir \
#    fastapi==0.115.11 \
#    uvicorn==0.34.0 \
#    pydantic==2.10.6 \
#    pydantic-settings==2.8.1 \
#    numpy==1.24.3 \
#    SQLAlchemy==2.0.39 \
#    python-dotenv==1.0.1

# # HTTP 및 통신 관련 패키지
# RUN pip install --no-cache-dir \
#    httpx==0.28.1 \
#    httpx-sse==0.4.0 \
#    aiohttp==3.11.14 \
#    requests==2.32.3 \
#    requests-toolbelt==1.0.0

# # AI/LLM 관련 패키지
# RUN pip install --no-cache-dir \
#    langchain==0.3.21 \
#    langchain-core==0.3.46 \
#    langchain-community==0.3.20 \
#    langchain-huggingface==0.1.2 \
#    langchain-cohere==0.4.3 \
#    langchain-openai==0.3.9 \
#    langchain-text-splitters==0.3.7 \
#    tiktoken==0.9.0 \
#    sentence-transformers==3.4.1 \
#    transformers==4.49.0 \
#    openai==1.67.0 \
#    cohere==5.14.0

# # CUDA 및 ML 관련 패키지
# RUN pip install --no-cache-dir \
#    torch==2.6.0 \
#    faiss-gpu==1.7.2 \
#    scikit-learn==1.6.1 \
#    scipy==1.15.2

# # llama-cpp-python 설치 (CUDA 활성화) - 수정된 부분
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/* && \
#     CMAKE_ARGS="-DLLAMA_CUBLAS=ON" FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python==0.3.8

# # 데이터베이스 관련
# RUN pip install --no-cache-dir \
#    mysql-connector-python==9.2.0

# # 유틸리티 패키지
# RUN pip install --no-cache-dir \
#    PyYAML==6.0.2 \
#    tqdm==4.67.1 \
#    pillow==11.1.0 \
#    orjson==3.10.15 \
#    dataclasses-json==0.6.7

# # 소스 코드 전체 복사
# COPY . .

# # 실행 스크립트에 실행 권한 부여
# RUN chmod +x /app/run_llm_service.sh /app/rag/run_api_server.sh

# # 포트 노출
# EXPOSE 8001

# # 컨테이너 실행 시, 두 스크립트를 백그라운드에서 동시에 실행하고 대기
# CMD ["/bin/sh", "-c", "/app/run_llm_service.sh & /app/rag/run_api_server.sh && wait"]
###############################################################################################
# FROM python:3.10-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y build-essential git && \
#     rm -rf /var/lib/apt/lists/*

# # 핵심 패키지만 설치
# RUN pip install --no-cache-dir \
#     fastapi==0.115.11 \
#     uvicorn==0.34.0 \
#     langchain==0.3.21 \
#     langchain-community==0.3.20 \
#     langchain-core==0.3.46 \
#     langchain-cohere==0.4.3 \
#     langchain-openai==0.3.9 \
#     tiktoken==0.9.0 \
#     faiss-cpu==1.10.0 \
#     python-dotenv==1.0.1 \
#     sentence-transformers==3.4.1 \
#     cohere==5.14.0 \
#     PyYAML==6.0.2

# COPY . .
# EXPOSE 8001

# # 서비스 시작
# CMD ["uvicorn", "rag.service_local:app", "--host", "0.0.0.0", "--port", "8001"]
########################################################################################
# FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# WORKDIR /app

# RUN apt-get update && apt-get install -y build-essential git python3 python3-pip && \
#     rm -rf /var/lib/apt/lists/*

# # 핵심 패키지만 설치
# RUN pip3 install --no-cache-dir \
#     fastapi==0.115.11 \
#     uvicorn==0.34.0 \
#     langchain==0.3.21 \
#     langchain-community==0.3.20 \
#     langchain-core==0.3.46 \
#     langchain-cohere==0.4.3 \
#     langchain-openai==0.3.9 \
#     tiktoken==0.9.0 \
#     faiss-cpu==1.10.0 \
#     python-dotenv==1.0.1 \
#     sentence-transformers==3.4.1 \
#     cohere==5.14.0 \
#     PyYAML==6.0.2

# COPY . .
# EXPOSE 8001

# # 서비스 시작
# CMD ["uvicorn", "rag.service_local:app", "--host", "0.0.0.0", "--port", "8001"]
########################################################################################
########################################################################################

# FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# WORKDIR /app

# # 기본 개발 도구 및 Python 설치
# RUN apt-get update && apt-get install -y build-essential git python3 python3-pip nvidia-cuda-toolkit && \
#     rm -rf /var/lib/apt/lists/*

# # 핵심 패키지 설치
# RUN pip3 install --no-cache-dir \
#     fastapi==0.115.11 \
#     uvicorn==0.34.0 \
#     langchain==0.3.21 \
#     langchain-community==0.3.20 \
#     langchain-core==0.3.46 \
#     langchain-cohere==0.4.3 \
#     langchain-openai==0.3.9 \
#     tiktoken==0.9.0 \
#     faiss-cpu==1.10.0 \
#     python-dotenv==1.0.1 \
#     sentence-transformers==3.4.1 \
#     cohere==5.14.0 \
#     PyYAML==6.0.2

# # llama-cpp-python 패키지 추가 (GPU 지원 포함)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON" pip3 install --no-cache-dir llama-cpp-python==0.3.8

# # CUDA 환경 변수 설정
# ENV PATH=/usr/local/cuda/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# COPY . .
# EXPOSE 8001

# # 수정된 실행 명령
# CMD ["python3", "-m", "uvicorn", "rag.service_local:app", "--host", "0.0.0.0", "--port", "8001"]
######################################################################################################
# CUDA 12.1 기반 런타임 이미지
# CUDA 11.8 기반 NVIDIA 공식 이미지 사용 (Python 3.10 포함)
# CUDA 11.8 기반 NVIDIA 공식 이미지 사용 (Python 3.10 포함)
# CUDA 11.8 기반 NVIDIA 공식 이미지 사용 (Python 3.10 포함)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 단계별 설치
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6"
RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-huggingface==0.1.0 langchain-cohere==0.3.0
RUN pip install --no-cache-dir sqlalchemy==2.0.35 tiktoken==0.7.0 requests==2.32.3

# CUDA 환경 설정 후 llama-cpp-python 설치
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
RUN pip install --no-cache-dir llama-cpp-python==0.2.91

RUN pip install --no-cache-dir accelerate==0.33.0 huggingface-hub==0.25.1 pydantic==1.10.13 python-dotenv==1.0.1 aiohttp==3.10.5

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["bash", "run_api_server.sh"]