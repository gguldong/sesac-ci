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
router = APIRouter()

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

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    # (기존 코드 유지)
    pass

# if __name__ == "__main__" 부분은 제거하거나 주석 처리