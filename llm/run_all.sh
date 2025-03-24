#!/bin/bash

# GPU 사용 여부 파라미터 처리
USE_GPU=1
if [ "$1" == "--cpu" ] || [ "$1" == "--no-gpu" ]; then
    USE_GPU=0
    echo "CPU 모드로 실행됩니다."
fi

echo "===== LLM 및 RAG 서비스 시작하기 ====="
cd "$(dirname "$0")"
BASEDIR="$(pwd)"

# 로그 디렉토리 생성
mkdir -p "$BASEDIR/logs"

# 가상환경 활성화 (프로젝트에 맞게 조정)
if [ -d "$BASEDIR/.llm_env" ]; then
    source $BASEDIR/.llm_env/bin/activate
    echo "가상환경 활성화: $BASEDIR/.llm_env"
elif [ -d "$WORKSPACE_DIR/.venv" ]; then
    source $WORKSPACE_DIR/.venv/bin/activate
    echo "가상환경 활성화: $WORKSPACE_DIR/.venv"
else
    echo "가상환경을 찾을 수 없습니다. 설치되어 있는지 확인하세요."
    exit 1
fi

# 필요한 패키지 확인 및 설치
echo "필요한 패키지 확인 중..."
pip install -q fastapi uvicorn httpx 
pip install -q faiss-cpu

# 1. RAG 서비스 실행 (백그라운드)
echo "1. RAG API 서버 시작 중..."
(cd $BASEDIR/rag && PYTHONPATH=$BASEDIR nohup python -m uvicorn api:app --host 0.0.0.0 --port 8001 > $BASEDIR/logs/rag_api.log 2>&1) &
RAG_PID=$!
echo "   RAG API 서버 시작됨 (PID: $RAG_PID)"
echo "   로그: $BASEDIR/logs/rag_api.log"

# 잠시 대기
sleep 2

# 2. LLM 서비스 실행 (백그라운드) - GPU 사용 여부에 따라 환경 변수 설정
echo "2. LLM 서비스 시작 중... (GPU 모드: $([ $USE_GPU -eq 1 ] && echo "활성화" || echo "비활성화"))"
(cd $BASEDIR && PYTHONPATH=$BASEDIR USE_GPU=$USE_GPU nohup python service.py > $BASEDIR/logs/llm_service.log 2>&1) &
LLM_PID=$!
echo "   LLM 서비스 시작됨 (PID: $LLM_PID)"
echo "   로그: $BASEDIR/logs/llm_service.log"

# PID 저장
echo "$RAG_PID" > $BASEDIR/rag_api.pid
echo "$LLM_PID" > $BASEDIR/llm_service.pid

echo
echo "===== 모든 서비스가 시작되었습니다 ====="
echo "서비스 종료 시: kill -9 $RAG_PID $LLM_PID"
echo "또는 ./stop_all.sh 실행"

# 종료 스크립트 생성
cat > $BASEDIR/stop_all.sh << EOL
#!/bin/bash
echo "모든 서비스 종료 중..."

if [ -f "$BASEDIR/rag_api.pid" ]; then
    RAG_PID=\$(cat "$BASEDIR/rag_api.pid")
    echo "RAG API 종료 중 (PID: \$RAG_PID)..."
    kill -9 \$RAG_PID 2>/dev/null || echo "이미 종료됨"
    rm "$BASEDIR/rag_api.pid"
fi

if [ -f "$BASEDIR/llm_service.pid" ]; then
    LLM_PID=\$(cat "$BASEDIR/llm_service.pid")
    echo "LLM 서비스 종료 중 (PID: \$LLM_PID)..."
    kill -9 \$LLM_PID 2>/dev/null || echo "이미 종료됨"
    rm "$BASEDIR/llm_service.pid"
fi

echo "모든 서비스가 종료되었습니다."
EOL

chmod +x $BASEDIR/stop_all.sh

echo
echo "RAG API: http://localhost:8001/api/rag"
echo "LLM API: http://localhost:8002/generate"
echo
echo "로그 확인:"
echo "tail -f $BASEDIR/logs/rag_api.log"
echo "tail -f $BASEDIR/logs/llm_service.log" 