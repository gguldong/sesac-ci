import os
from dotenv import load_dotenv
import httpx
import json
import asyncio

load_dotenv()

# LLM 서비스 URL 설정
LLM_SERVICE_URL = "http://llm-service:8001/generate"

async def request_openchat(message: str, conversation_history: str = "", user_info: dict[str, str] = {}, rag_context: str = ""):
    try:
        # LLM 서비스 API 요청 데이터 준비
        request_data = {
            "message": message,
            "conversation_history": conversation_history,
            "user_info": user_info,
            "rag_context": rag_context
        }
        
        # LLM 서비스 API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                LLM_SERVICE_URL,
                json=request_data,
                timeout=1000  # 타임아웃 설정 (초)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                error_message = f"LLM 서비스 오류: 상태 코드 {response.status_code}"
                print(error_message)
                return "죄송합니다. 서비스 연결에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
                
    except Exception as e:
        error_message = f"OpenChat API 호출 중 에러 발생: {str(e)}"
        print(error_message)
        return "죄송합니다. 서비스 연결에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요." 