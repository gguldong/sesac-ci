from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
from typing import Any, Dict, List, Union
import logging
import traceback
import requests
import json
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vllm_client.log")
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# VLLM 서버 설정
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "https://qnsqwbfncpfgeqxe.tunnel-pt.elice.io")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    rag_context: Any = None  # 문자열 또는 딕셔너리 형태 모두 허용

class ChatResponse(BaseModel):
    response: str

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        start_time = time.time()
        logger.info(f"요청 수신: 메시지 길이 {len(request.message)}, 대화 기록 길이 {len(request.conversation_history)}")
        
        # 메시지 배열 준비
        messages = []
        valid_roles = {"system", "assistant222", "user111", "function", "tool", "developer"}
        
        # 시스템 프롬프트 설정
        system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다. 
사용자의 질문에 대해 명확하고 정확하게 답변해주세요.
정책 정보가 제공된 경우, 해당 정보를 바탕으로 답변해주세요."""

        rag_context = request.rag_context
        # 너무 길면 추가 제한
        max_chars = 25000  # 약 3000 토큰에 해당
        if len(rag_context) > max_chars:
            trimmed_rag_context = trimmed_rag_context[:max_chars] + "...(내용 일부 생략)"
        
        logger.info(f"처리된 RAG 컨텍스트 길이: {len(trimmed_rag_context)}")
            
        system_prompt = """당신은 정책 전문가입니다. 주어진 정책 정보를 바탕으로 사용자의 질문에 답변해주세요.
답변은 친절하고 명확하게 해주세요. 정책 정보에 없는 내용은 생성하지 마세요."""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "system", "content": trimmed_rag_context})
 

        # 대화 기록 처리
        if request.conversation_history:
            lines = request.conversation_history.strip().split('\n')
            for line in lines:
                if line.startswith("사용자 정보:"):
                    messages.append({"role": "system", "content": line})
                else:
                    try:
                        role, content = line.split("::: ", 1)
                        if role not in valid_roles:
                            messages.append({"role": "user", "content": line})
                        else:
                            if role == "user111":
                                role = "user"
                            elif role == "assistant222":
                                role = "assistant"
                            messages.append({"role": role, "content": content})
                    except ValueError:
                        messages.append({"role": "user", "content": line})

        # 사용자 메시지 추가
        messages.append({"role": "user", "content": request.message})

        # OpenChat 포맷으로 메시지 변환
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"GPT4 Correct User: {msg['content']}<|end_of_turn|>"
            elif msg["role"] == "system":
                prompt += f"GPT4 Correct User: <system>{msg['content']}</system><|end_of_turn|>"
            else:
                prompt += f"GPT4 Correct Assistant: {msg['content']}<|end_of_turn|>"
        
        prompt += "GPT4 Correct Assistant:"
        
        # 프롬프트 텍스트 길이 로깅
        logger.info(f"최종 프롬프트 길이: {len(prompt)}")
        
        # VLLM 서버에 API 요청
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 500,
            "stop": ["<|end_of_turn|>"]
        }
        VLLM_GENERATE = VLLM_SERVER_URL + "/generate"
        logger.info(f"외부 VLLM 서버 요청 시작: {VLLM_GENERATE}")
        
        try:
            response = requests.post(
                VLLM_GENERATE,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            
            # 응답 처리
            response_data = response.json()
            logger.debug(f"VLLM 서버 응답: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
            # 응답 구조에 따라 텍스트 추출 (API 응답 구조에 맞게 수정 필요)
            if "choices" in response_data and len(response_data["choices"]) > 0:
                generated_text = response_data["choices"][0]["text"].strip()
            else:
                raise ValueError("VLLM 서버 응답에서 생성된 텍스트를 찾을 수 없습니다")
            
            end_time = time.time()
            logger.info(f"응답 완료: 응답 길이 {len(generated_text)}, 처리 시간: {end_time - start_time:.2f}초")
            
            return ChatResponse(response=generated_text)
            
        except requests.exceptions.RequestException as e:
            error_message = f"VLLM 서버 통신 오류: {str(e)}"
            logger.error(error_message)
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"서버 오류 응답: {e.response.text}")
            raise HTTPException(status_code=503, detail=error_message)

    except Exception as e:
        error_message = f"요청 처리 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/health")
async def health_check():
    """서비스 상태 확인용 엔드포인트"""
    try:
        # VLLM 서버 연결 상태 확인
        headers = {"Content-Type": "application/json"}
        
        # 간단한 헬스 체크용 요청
        response = requests.post(
            VLLM_SERVER_URL,
            headers=headers,
            json={
                "prompt": "Hello",
                "max_tokens": 1
            },
            timeout=5  # 짧은 타임아웃으로 설정
        )
        response.raise_for_status()
        
        return {
            "status": "healthy", 
            "vllm_server": "connected",
            "vllm_server_url": VLLM_SERVER_URL
        }
    except Exception as e:
        logger.warning(f"헬스 체크 중 오류: {str(e)}")
        return {
            "status": "degraded",
            "vllm_server": "disconnected",
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("SERVICE_PORT", "8002"))
    logger.info(f"서비스 시작 - 포트: {port}, VLLM 서버: {VLLM_SERVER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)