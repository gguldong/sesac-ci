#!/bin/bash

echo "RAG API 서버 시작..."
cd "$(dirname "$0")"

# 가상환경 활성화 (가상환경 이름이 다르면 수정 필요)
if [ -d "../.llm_env" ]; then
    source ../.llm_env/bin/activate
    echo "가상환경 활성화: ../.llm_env"
elif [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
    echo "가상환경 활성화: ../../.venv"
else
    echo "가상환경을 찾을 수 없습니다. 설치되어 있는지 확인하세요."
fi

# 필요한 패키지 확인 및 설치
pip install -q fastapi uvicorn

# API 서버 실행
echo "FastAPI 서버 실행 중..."
python -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload