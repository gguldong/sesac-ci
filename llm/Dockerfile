# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     git \
#     cmake \
#     g++ \
#     make \
#     libopenblas-dev \
#     && rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python3 /usr/bin/python

# RUN pip install --upgrade pip setuptools wheel

# RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# # 단계별 설치
# RUN pip install --no-cache-dir numpy==1.26.4
# RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6"
# RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-huggingface==0.1.0 langchain-cohere==0.3.0
# RUN pip install --no-cache-dir sqlalchemy==2.0.35 tiktoken==0.7.0 requests==2.32.3

# # CUDA 환경 설정 후 llama-cpp-python 설치 (사전 컴파일된 휠 사용)
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# RUN pip install --no-cache-dir accelerate==0.33.0 huggingface-hub==0.25.1 pydantic==1.10.13 python-dotenv==1.0.1 aiohttp==3.10.5

# COPY . .

# ENV PYTHONUNBUFFERED=1

# EXPOSE 8001

# CMD ["bash", "/app/rag/run_api_server.sh"]
###########################################################################################################
###########################################################################################################
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# WORKDIR /app

# # 필수 시스템 패키지 설치
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     git \
#     cmake \
#     g++ \
#     make \
#     libopenblas-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Python 3를 기본 python으로 설정
# RUN ln -s /usr/bin/python3 /usr/bin/python

# # pip 업그레이드
# RUN pip install --upgrade pip setuptools wheel

# # PyTorch 설치 (CUDA 11.8 호환)
# RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# # 단계별 패키지 설치
# RUN pip install --no-cache-dir numpy==1.26.4
# RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6"
# # langchain과 pydantic v2를 함께 설치하여 호환성 보장
# RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-huggingface==0.1.0 langchain-cohere==0.3.0 pydantic==2.9.2
# RUN pip install --no-cache-dir sqlalchemy==2.0.35 tiktoken==0.7.0 requests==2.32.3

# # CUDA 환경 설정 후 llama-cpp-python 설치 (사전 컴파일된 휠 사용)
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# # 나머지 패키지 설치 (pydantic은 위에서 이미 설치됨)
# RUN pip install --no-cache-dir accelerate==0.33.0 huggingface-hub==0.25.1 python-dotenv==1.0.1 aiohttp==3.10.5

# # 소스 코드 복사
# COPY . .

# # 환경 변수 설정
# ENV PYTHONUNBUFFERED=1

# # 포트 노출
# EXPOSE 8001

# # 실행 명령
# CMD ["bash", "/app/rag/run_api_server.sh"]

###########################################################################################################
###########################################################################################################
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# WORKDIR /app

# # 필수 시스템 패키지 설치
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     git \
#     cmake \
#     g++ \
#     make \
#     libopenblas-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Python 3를 기본 python으로 설정
# RUN ln -s /usr/bin/python3 /usr/bin/python

# # pip 업그레이드
# RUN pip install --upgrade pip setuptools wheel

# # PyTorch 설치 (CUDA 11.8 호환)
# RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# # 단계별 패키지 설치
# RUN pip install --no-cache-dir numpy==1.26.4
# RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6"

# # transformers와 sentence-transformers를 먼저 설치하여 의존성 충돌 방지
# # huggingface-hub 버전을 0.26.0으로 업그레이드
# RUN pip install --no-cache-dir transformers==4.45.2 huggingface-hub==0.26.0 sentence-transformers==3.1.1

# # langchain과 관련 패키지 설치
# # pydantic은 langchain과 호환되는 버전으로 유지 (2.9.2는 최신 버전으로 문제 없음)
# RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-huggingface==0.1.0 langchain-cohere==0.3.0 pydantic==2.9.2

# # 나머지 패키지 설치
# RUN pip install --no-cache-dir sqlalchemy==2.0.35 tiktoken==0.7.0 requests==2.32.3

# # CUDA 환경 설정 후 llama-cpp-python 설치 (사전 컴파일된 휠 사용)
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# # 나머지 패키지 설치
# # huggingface-hub은 위에서 이미 설치됨
# RUN pip install --no-cache-dir accelerate==0.33.0 python-dotenv==1.0.1 aiohttp==3.10.5

# # 소스 코드 복사
# COPY . .

# # 환경 변수 설정
# ENV PYTHONUNBUFFERED=1

# # 포트 노출
# EXPOSE 8001

# # 실행 명령
# CMD ["bash", "/app/rag/run_api_server.sh"]
#############################################################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3를 기본 python으로 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 11.8 호환)
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 단계별 패키지 설치
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6"

# transformers와 sentence-transformers를 먼저 설치하여 의존성 충돌 방지
# huggingface-hub 버전을 0.26.0으로 업그레이드
RUN pip install --no-cache-dir transformers==4.45.2 huggingface-hub==0.26.0 sentence-transformers==3.1.1

# langchain과 관련 패키지 설치
# pydantic은 langchain과 호환되는 버전으로 유지 (2.9.2는 최신 버전으로 문제 없음)
RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-huggingface==0.1.0 langchain-cohere==0.3.0 pydantic==2.9.2

# 나머지 패키지 설치
RUN pip install --no-cache-dir sqlalchemy==2.0.35 tiktoken==0.7.0 requests==2.32.3

# CUDA 환경 설정 후 llama-cpp-python 설치 (사전 컴파일된 휠 사용)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# FAISS-GPU 설치 (CUDA 11.8 호환)
RUN pip install --no-cache-dir faiss-gpu==1.7.2

# 나머지 패키지 설치
# huggingface-hub은 위에서 이미 설치됨
RUN pip install --no-cache-dir accelerate==0.33.0 python-dotenv==1.0.1 aiohttp==3.10.5 uvloop==0.19.0

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8001

# 실행 명령
CMD ["bash", "/app/rag/run_api_server.sh"]