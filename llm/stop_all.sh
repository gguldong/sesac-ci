#!/bin/bash
echo "모든 서비스 종료 중..."

if [ -f "/home/elicer/Fundit_2.1/LLM/rag_api.pid" ]; then
    RAG_PID=$(cat "/home/elicer/Fundit_2.1/LLM/rag_api.pid")
    echo "RAG API 종료 중 (PID: $RAG_PID)..."
    kill -9 $RAG_PID 2>/dev/null || echo "이미 종료됨"
    rm "/home/elicer/Fundit_2.1/LLM/rag_api.pid"
fi

if [ -f "/home/elicer/Fundit_2.1/LLM/llm_service.pid" ]; then
    LLM_PID=$(cat "/home/elicer/Fundit_2.1/LLM/llm_service.pid")
    echo "LLM 서비스 종료 중 (PID: $LLM_PID)..."
    kill -9 $LLM_PID 2>/dev/null || echo "이미 종료됨"
    rm "/home/elicer/Fundit_2.1/LLM/llm_service.pid"
fi

echo "모든 서비스가 종료되었습니다."
