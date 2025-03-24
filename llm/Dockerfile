# CUDA 12.4 기반 devel 이미지 (Ubuntu 22.04) 사용 → GPU 지원 필요 시 사용
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
    nvidia-driver-535 \
    && rm -rf /var/lib/apt/lists/*

# pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Python 가상환경 생성 및 활성화
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CUDA 빌드를 위한 환경 변수 설정 (GPU 지원 활성화)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc"
ENV PATH=/usr/local/cuda-12.4/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt ./ 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# llama-cpp-python 설치
RUN pip install llama-cpp-python --force-reinstall --no-cache-dir

# faiss-gpu를 CUDA 지원 버전으로 강제 재설치 (Python 3.10용)
RUN pip install --force-reinstall --no-cache-dir faiss-gpu==1.7.2 --extra-index-url https://pypi.nvidia.com

# 소스 코드 전체 복사
COPY . .

# 실행 스크립트에 실행 권한 부여
RUN chmod +x /app/run_llm_service.sh /app/rag/run_api_server.sh

# 컨테이너 실행 시, 두 스크립트를 백그라운드에서 동시에 실행하고 대기
CMD ["/bin/sh", "-c", "/app/run_llm_service.sh & /app/rag/run_api_server.sh && wait"]
