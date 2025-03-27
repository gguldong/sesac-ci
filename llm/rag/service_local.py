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

# app = APIRouter()
# 윗줄 한 줄에서 아래 4줄로 수정 코드
from fastapi import FastAPI
router = APIRouter()
app = FastAPI()
app.include_router(router)




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

# 시스템 환경에 따라 GPU 사용 여부 결정
def check_cuda_available(model_path):
    # 환경 변수로 강제 비활성화된 경우
    if not USE_GPU:
        print("환경 변수에 따라 CPU 모드로 실행됩니다.")
        return False
    
    # CUDA 사용 가능 여부 확인 (여러 방법 시도)
    cuda_available = False
    
    # 방법 1: subprocess로 nvidia-smi 명령어 실행
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        if result.returncode == 0:
            print("NVIDIA GPU 감지됨 (nvidia-smi 확인)")
            cuda_available = True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi 실행 실패: NVIDIA 드라이버가 설치되지 않았거나 접근할 수 없습니다.")
    
    # # 방법 2: ctypes로 CUDA 라이브러리 로드 시도
    # if not cuda_available:
    #     try:
    #         import ctypes
    #         ctypes.CDLL("libcudart.so")
    #         print("CUDA 라이브러리 로드됨")
    #         cuda_available = True
    #     except OSError:
    #         print("CUDA 라이브러리를 로드할 수 없습니다.")
    
    # 방법 3: llama-cpp-python의 CUDA 지원 확인
    if cuda_available:
        try:
            # 간단한 Llama 객체 생성 시도 (모델 로드 없이)
            from llama_cpp import Llama
            test_llama = Llama(
                model_path=model_path,
                n_gpu_layers=1,
                verbose=True,
                n_ctx=512,
                seed=-1,
                use_mmap=False,
                use_mlock=False
            )
            # 오류가 발생하지 않았다면 CUDA 지원이 있는 것
            print("llama-cpp-python이 CUDA 지원과 함께 설치됨")
        except Exception as e:
            print(f"llama-cpp-python CUDA 지원 확인 실패: {str(e)}")
            # GPU를 감지했지만 llama-cpp-python이 CUDA를 지원하지 않는 경우
            print("llama-cpp-python이 CUDA 지원 없이 설치되었을 수 있습니다.")
            print("CUDA 지원을 위해 다음 명령어로 재설치하세요:")
            print("pip uninstall -y llama-cpp-python")
            print("CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
            cuda_available = False
    
    if cuda_available:
        print("CUDA 사용 가능 - GPU 모드로 실행됩니다.")
    else:
        print("CUDA 사용 불가능 - CPU 모드로 실행됩니다.")
    
    return cuda_available

# 모델 초기화
# 주석에 표시된 상대경로 사용
# model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
#                          "OpenChat-3.5-7B-Mixtral-v2.0.i1-Q4_K_M.gguf")
model_path = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/OpenChat-3.5-7B-Mixtral-v2.0.i1-Q4_K_M.gguf"))
#이렇게 수정하면 모델 파일이 호스트의 PVC 경로에서 컨테이너의 /app/models 경로로 마운트되어 서비스가 모델 파일에 접근할 수 있게 됩니다.

print(f"모델 경로: {model_path}")

# GPU 사용 가능 여부 확인
CUDA_AVAILABLE = check_cuda_available(model_path)



# 모델 로드 (서버 시작 시 한 번만 로드됨)
llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1 if CUDA_AVAILABLE else 0,  # CUDA 사용 가능하면 모든 레이어를 GPU에서 실행, 아니면 CPU만 사용
    verbose=True      # 로딩 시 CUDA 사용 여부 로그 출력
)

print(f"모델 로드 완료 - GPU 사용: {CUDA_AVAILABLE}")

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    user_info: Dict[str, str] = []
    rag_context: str = None  # 문자열 또는 딕셔너리 형태 모두 허용

class ChatResponse(BaseModel):
    response: str

@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        # 메시지 배열 준비
        start_time = time.time()
        logger.info(f"요청 수신: 메시지 길이 {len(request.message)}, 대화 기록 길이 {len(request.conversation_history)}")
        
        # 메시지 배열 준비
        messages = []
        valid_roles = {"system", "assistant222", "user111", "function", "tool", "developer"}
        rag_context = request.rag_context or ""
            
        system_prompt = f"""당신은 사용자 맞춤형 정부 정책 추천 시스템입니다.

사용자 정보:
- 성별: {request.user_info.get("gender", "미지정")}
- 지역: {request.user_info.get("area", "미지정")}
- 특성: {request.user_info.get("personalCharacteristics", "미지정")}

아래는 이 사용자에게 적합한 보조금/지원 정책 요약 리스트입니다:

{rag_context}

이 정책들을 기반으로, 아래와 같은 형식으로 **전체 사용자 맞춤 요약**을 작성하세요:

1. 이 사용자가 어떤 생활환경 또는 상황에 처해있다고 판단되는지 (ex: 저소득 맞벌이 가구, 어린 자녀가 있는 가정 등)
2. 이러한 상황에 왜 위 정책들이 도움이 될 수 있는지 설명
3. 이 중 몇몇 정책은 **최대 얼마까지** 지원이 가능한지 언급
4. 마지막 "자세한 내용은 각 정책의 링크를 참고해 주세요." 이 내용 추가

모든 내용은 총 5~8문장으로 구성하고, 정책 하나만 말하지 말고 **전체를 종합적으로 설명**하세요."""

        
        system_message = f"{system_prompt}"
        messages = [{"role": "system", "content": system_message}]
        print(f"system_message: {system_message}")

        # messages.append({"role": "system", "content": system_prompt})
        # messages.append({"role": "system", "content": rag_context})
 

        # 대화 기록 처리
        if request.conversation_history:
            lines = request.conversation_history.strip().split('\n')
            for line in lines:
                if line.startswith("사용자 정보:"):
                    messages.append({"role": "system", "content": line})
                elif line.strip():
                    try:
                        role, content = line.split("::: ", 1)
                        role = "user" if role == "user111" else "assistant" if role == "assistant222" else role
                        if role in {"user", "assistant"}:
                            messages.append({"role": role, "content": content})
                        else:
                            messages.append({"role": "user", "content": line})
                    except ValueError:
                        messages.append({"role": "user", "content": line})

        # 사용자 메시지 추가
        messages.append({"role": "user", "content": request.message})

        # OpenChat 포맷으로 변환
        prompt = "<|system|>" + messages[0]["content"] + "<|end_of_turn|>"
        for msg in messages[1:]:
            if msg["role"] == "user":
                prompt += f"<|user|>{msg['content']}<|end_of_turn|>"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>{msg['content']}<|end_of_turn|>"
        prompt += "<|assistant|>"

        # 토큰 수 체크 (tiktoken 필요)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(prompt)
        if len(tokens) > 8192:
            prompt = encoding.decode(tokens[:8192-500]) + "...(생략됨)"
        
        logger.info(f"최종 프롬프트 길이: {len(prompt)}, 토큰 수: {len(tokens)}")
        logger.debug(f"프롬프트: {prompt}")
        
        # 토큰 디버깅 정보 출력
        max_tokens = 500  # 초기값
        try:
            # 프롬프트의 토큰 수 계산 (tiktoken 사용)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(prompt)
            token_count = len(tokens)
            ctx_window = llm.n_ctx() if callable(llm.n_ctx) else llm.n_ctx
            print(f"총 토큰 수: {token_count}, 컨텍스트 윈도우: {ctx_window}")

            # 동적으로 max_tokens 계산
            buffer = 500  # 응답 여유 공간
            max_tokens = max(1000, ctx_window - token_count - buffer)  # 최소 1000 토큰 보장
            if token_count > ctx_window - 300:
                raise HTTPException(status_code=400, detail="입력 토큰이 컨텍스트 윈도우를 초과했습니다.")
        except Exception as e:
            print(f"토큰 계산 중 오류: {str(e)}")
        print(f"설정된 max_tokens: {max_tokens}")
        
        # OpenChat 모델로 응답 생성
        output = llm(prompt, max_tokens=max_tokens, temperature=0.2)
        openchat_response = output["choices"][0]["text"]
        
        return ChatResponse(response=openchat_response)

    except Exception as e:
        error_message = f"모델 추론 중 오류 발생: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8002) 