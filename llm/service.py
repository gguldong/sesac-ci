from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from llama_cpp import Llama
import os
from typing import Any, Dict, List, Union

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# GPU 사용 여부 확인 및 설정
USE_GPU = os.environ.get("USE_GPU", "1").lower() in ("1", "true", "yes", "y")

# 시스템 환경에 따라 GPU 사용 여부 결정
def check_cuda_available():
    # 환경 변수로 강제 비활성화된 경우
    if not USE_GPU:
        print("환경 변수에 따라 CPU 모드로 실행됩니다.")
        return False
    else:
        print("CUDA")
        return True 
  

# GPU 사용 가능 여부 확인
CUDA_AVAILABLE = check_cuda_available()

# 모델 초기화
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "OpenChat-3.5-7B-Mixtral-v2.0.i1-Q4_K_M.gguf")

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
    rag_context: Any = None  # 문자열 또는 딕셔너리 형태 모두 허용

class ChatResponse(BaseModel):
    response: str

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        # 메시지 배열 준비
        messages = []
        valid_roles = {"system", "assistant222", "user111", "function", "tool", "developer"}
        
        # 시스템 프롬프트 설정
        system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다. 
사용자의 질문에 대해 명확하고 정확하게 답변해주세요.
정책 정보가 제공된 경우, 해당 정보를 바탕으로 답변해주세요."""

        # RAG 컨텍스트가 있으면 정책 전문가 프롬프트로 변경
        if request.rag_context:
            processed_rag_context = []
            
            # 딕셔너리 형태로 전달된 경우 (새로운 API 응답 구조)
            if isinstance(request.rag_context, dict):
                # 문서 정보 추출 (최대 15개까지만)
                documents = request.rag_context.get("documents", {})
                sources = request.rag_context.get("sources", [])
                
                # 소스 목록으로 컨텍스트 구성
                doc_count = 0
                for source in sources:
                    if doc_count >= 15:
                        break
                    
                    doc_count += 1
                    title = source.get("title", "제목 없음")
                    content = source.get("content", "내용 없음")
                    service_id = source.get("service_id", "ID 없음")
                    eligibility = source.get("eligibility", "자격 요건 정보 없음")
                    benefits = source.get("benefits", "혜택 정보 없음")
                    
                    # 문서 형식으로 추가
                    processed_rag_context.append(f"문서: {title}")
                    processed_rag_context.append(f"내용: {content}")
                    if eligibility:
                        processed_rag_context.append(f"자격 요건: {eligibility}")
                    if benefits:
                        processed_rag_context.append(f"혜택: {benefits}")
                    processed_rag_context.append("")  # 문서 구분을 위한 빈 줄
                
                # 문서가 없거나 소스가 없는 경우 documents에서 직접 추출
                if doc_count == 0 and documents:
                    for doc_id, doc_info in documents.items():
                        if doc_count >= 15:
                            break
                        
                        doc_count += 1
                        content = doc_info.get("내용", "내용 없음")
                        metadata = doc_info.get("메타데이터", {})
                        title = metadata.get("title", "제목 없음")
                        
                        # 문서 형식으로 추가
                        processed_rag_context.append(f"문서: {title}")
                        processed_rag_context.append(f"내용: {content}")
                        
                        # 메타데이터에서 추가 정보 추출
                        if "eligibility" in metadata:
                            processed_rag_context.append(f"자격 요건: {metadata['eligibility']}")
                        if "benefits" in metadata:
                            processed_rag_context.append(f"혜택: {metadata['benefits']}")
                        
                        processed_rag_context.append("")  # 문서 구분을 위한 빈 줄
                
                # 서비스 ID 정보 추가
                service_ids = request.rag_context.get("service_ids", [])
                common_ids = request.rag_context.get("common_ids", [])
                if service_ids or common_ids:
                    id_summary = "관련 서비스 ID: " + ", ".join(service_ids if service_ids else common_ids)
                    processed_rag_context.append(id_summary)
            
            # 문자열 형태로 전달된 경우 (기존 방식)
            elif isinstance(request.rag_context, str):
                # 상위 15개 문서로 제한하고 문서 ID 제거
                rag_context_lines = request.rag_context.strip().split('\n')
                doc_count = 0
                current_doc_lines = []
                in_document = False
                
                for line in rag_context_lines:
                    # 새 문서 시작
                    if line.startswith("문서 ID:") or line.startswith("문서:") or line.startswith("==="):
                        # 이전 문서 처리
                        if in_document and current_doc_lines:
                            # 긴 문서인 경우 내용 요약 (첫 5줄과 마지막 5줄만 유지)
                            if len(current_doc_lines) > 15:
                                top_lines = current_doc_lines[:5]
                                bottom_lines = current_doc_lines[-5:]
                                current_doc_lines = top_lines + ["..."] + bottom_lines
                            
                            processed_rag_context.extend(current_doc_lines)
                            processed_rag_context.append("")  # 문서 구분을 위한 빈 줄
                        
                        # 새 문서 시작
                        doc_count += 1
                        if doc_count > 15:
                            break
                        current_doc_lines = []
                        in_document = True
                        continue
                    
                    # ID 패턴 제거 (예: "문서 ID: 12345" -> 제거)
                    if "ID:" in line or "번호:" in line:
                        continue
                    
                    # 중요하지 않은 메타데이터 필터링
                    if any(skip in line for skip in ["생성일:", "수정일:", "작성자:", "버전:", "URL:"]):
                        continue
                    
                    # 현재 문서 내용 저장
                    if in_document:
                        current_doc_lines.append(line)
                
                # 마지막 문서 처리
                if in_document and current_doc_lines:
                    # 긴 문서인 경우 내용 요약 (첫 5줄과 마지막 5줄만 유지)
                    if len(current_doc_lines) > 15:
                        top_lines = current_doc_lines[:5]
                        bottom_lines = current_doc_lines[-5:]
                        current_doc_lines = top_lines + ["..."] + bottom_lines
                    
                    processed_rag_context.extend(current_doc_lines)
            
            # 처리된 컨텍스트로 변환
            trimmed_rag_context = "\n".join(processed_rag_context)
            
            # 너무 길면 추가 제한
            max_chars = 12000  # 약 3000 토큰에 해당
            if len(trimmed_rag_context) > max_chars:
                trimmed_rag_context = trimmed_rag_context[:max_chars] + "...(내용 일부 생략)"
            
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
        
        # 토큰 디버깅 정보 출력
        try:
            tokens = llm.tokenize(prompt.encode('utf-8'))
            token_count = len(tokens)
            print(f"총 토큰 수: {token_count}, 컨텍스트 윈도우: {llm.n_ctx}")
            
            if token_count > llm.n_ctx - 500:  # 응답 토큰을 위한 여유 공간 확보
                print("경고: 토큰 수가 컨텍스트 윈도우에 가깝습니다. RAG 컨텍스트를 더 줄여야 할 수 있습니다.")
        except Exception as e:
            print(f"토큰 계산 중 오류: {str(e)}")
        
        # OpenChat 모델로 응답 생성
        output = llm(prompt, max_tokens=500, temperature=0.7)
        openchat_response = output["choices"][0]["text"]
        
        return ChatResponse(response=openchat_response)

    except Exception as e:
        error_message = f"모델 추론 중 오류 발생: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 