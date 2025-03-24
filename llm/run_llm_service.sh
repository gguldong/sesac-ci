#!/bin/bash

# GPU 사용 여부 파라미터 처리
USE_GPU=0
if [ "$1" == "--cpu" ] || [ "$1" == "--no-gpu" ]; then
    USE_GPU=0
    echo "CPU 모드로 실행됩니다."
elif [ "$1" == "--gpu" ]; then
    USE_GPU=1
    echo "GPU 모드로 실행됩니다 (사용 가능한 경우)."
fi

echo "LLM 서비스 시작..."
cd "$(dirname "$0")"

# 가상환경 활성화
if [ -d ".llm_env" ]; then
    source .llm_env/bin/activate
    echo "가상환경 활성화: .llm_env"
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "가상환경 활성화: ../.venv"
# else
#     echo "가상환경을 찾을 수 없습니다. 설치되어 있는지 확인하세요."
fi

# 필요한 패키지 확인 및 설치
pip install -q fastapi uvicorn

# 로그 디렉토리 생성
mkdir -p logs

# 서비스 실행 (GPU 옵션 적용)
echo "LLM 서비스 실행 중... (GPU 모드: $([ $USE_GPU -eq 1 ] && echo "활성화" || echo "비활성화"))"
export USE_GPU=$USE_GPU
python service.py

# 백그라운드 실행이 필요한 경우 아래 코드 사용
# (export USE_GPU=$USE_GPU; nohup python service.py > logs/llm_service.log 2>&1) &
# echo "LLM 서비스가 백그라운드에서 실행 중입니다. 로그: logs/llm_service.log" 