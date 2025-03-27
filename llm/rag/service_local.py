from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from llama_cpp import Llama
import os
from typing import Any, Dict, List, Union
import logging
import time
import tiktoken

# 환경 변수 로드
load_dotenv()

# APIRouter 객체 생성
llm_router = APIRouter()

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

# GPU 사용 여부 확인 및 설정
USE_GPU = os.environ.get("USE_GPU", "1").lower() in ("1", "true", "yes", "y")

# (중략: check_cuda_available, 모델 로드 등 기존 코드 유지)
# GPU 사용 여부 확인 함수
def check_cuda_available(model_path):
    """CUDA 사용 가능 여부를 확인하는 함수"""
    try:
        import torch
        if torch.cuda.is_available():
            if os.path.exists(model_path):
                return True
            else:
                logger.warning(f"모델 파일이 존재하지 않습니다: {model_path}")
                return False
        else:
            logger.warning("CUDA를 사용할 수 없습니다: GPU 지원이 없거나 CUDA가 설치되지 않았습니다")
            return False
    except ImportError:
        logger.warning("PyTorch가 설치되지 않았습니다 - GPU를 사용할 수 없습니다")
        return False
    except Exception as e:
        logger.warning(f"CUDA 확인 중 오류 발생: {str(e)}")
        return False



# 모델 초기화
model_path = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/OpenChat-3.5-7B-Mixtral-v2.0.i1-Q4_K_M.gguf"))
CUDA_AVAILABLE = check_cuda_available(model_path)

llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1 if CUDA_AVAILABLE else 0,
    verbose=True
)

print(f"모델 로드 완료 - GPU 사용: {CUDA_AVAILABLE}")

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    user_info: Dict[str, str] = []
    rag_context: str = None

class ChatResponse(BaseModel):
    response: str

@llm_router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        start_time = time.time()
        logger.info(f"요청 수신: 메시지 길이 {len(request.message)}, 대화 기록 길이 {len(request.conversation_history)}")
        
        # 메시지 배열 준비
        messages = []
        valid_roles = {"system", "assistant", "user", "function", "tool", "developer"}
        
        # 시스템 프롬프트 설정
        system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다. 
사용자의 질문에 대해 명확하고 정확하게 답변해주세요.
정책 정보가 제공된 경우, 해당 정보를 바탕으로 답변해주세요."""

        # RAG 컨텍스트 처리
        rag_context = request.rag_context
        if rag_context:
            # 너무 길면 추가 제한
            max_chars = 25000  # 약 3000 토큰에 해당
            trimmed_rag_context = rag_context
            if len(rag_context) > max_chars:
                trimmed_rag_context = rag_context[:max_chars] + "...(내용 일부 생략)"
            
            logger.info(f"처리된 RAG 컨텍스트 길이: {len(trimmed_rag_context)}")
                
            system_prompt = """당신은 정책 전문가입니다. 주어진 정책 정보를 바탕으로 사용자의 질문에 답변해주세요.
답변은 친절하고 명확하게 해주세요. 정책 정보에 없는 내용은 생성하지 마세요."""
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "system", "content": trimmed_rag_context})
        else:
            messages.append({"role": "system", "content": system_prompt})

        # 대화 기록 처리
        if request.conversation_history:
            lines = request.conversation_history.strip().split('\n')
            for line in lines:
                if line.startswith("사용자 정보:"):
                    messages.append({"role": "system", "content": line})
                else:
                    try:
                        if "::: " in line:
                            role, content = line.split("::: ", 1)
                            if role not in valid_roles:
                                messages.append({"role": "user", "content": line})
                            else:
                                messages.append({"role": role, "content": content})
                        else:
                            messages.append({"role": "user", "content": line})
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
        
        # llama-cpp를 사용해 직접 추론
        result = llm(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500,
            stop=["<|end_of_turn|>"]
        )
        
        # 응답 추출
        generated_text = result["choices"][0]["text"].strip()
        
        end_time = time.time()
        logger.info(f"응답 완료: 응답 길이 {len(generated_text)}, 처리 시간: {end_time - start_time:.2f}초")
        
        return ChatResponse(response=generated_text)
            
    except Exception as e:
        error_message = f"요청 처리 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_message)