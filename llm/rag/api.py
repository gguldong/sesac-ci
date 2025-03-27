from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import Dict, List, Any, Optional, Union
from main import main
from service_local import llm_router  # 같은 디렉토리에 있으므로 직접 임포트

app = FastAPI(title="RAG API Service", description="AI 시스템을 위한 API 서비스")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 출처만 허용하도록 변경해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG API용 라우터 생성
rag_router = APIRouter()  # 새 라우터 생성


# # 새 라우터 생성 - RAG API용
# router = APIRouter()  # 이 부분 추가 - RAG 엔드포인트를 위한 별도 라우터


# LLM 라우터 추가 - '/llm' 경로에 마운트
app.include_router(llm_router, tags=["LLM Service"])

# RAG 라우터 추가
app.include_router(rag_router, tags=["RAG Service"])  # 이 부분 추가 - router를 app에 포함




class RagRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class RagResponse(BaseModel):
    documents: Dict[str, Dict[str, Any]]
    common_ids: List[str]
    vector_only_ids: List[str] 
    sql_only_ids: List[str]
    sources: Optional[List[Dict[str, Any]]] = None
    service_ids: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "documents": {
                    "문서번호 1": {
                        "내용": "청년 주거 지원 정책...",
                        "메타데이터": {"service_id": "001", "title": "청년 주거 지원"}
                    }
                },
                "common_ids": ["001", "002"],
                "vector_only_ids": ["003"],
                "sql_only_ids": ["004"],
                "sources": [
                    {
                        "title": "청년 주거 지원 정책",
                        "content": "정책 내용...",
                        "service_id": "001",
                        "eligibility": "20-39세 청년",
                        "benefits": "월 50만원 주거비 지원"
                    }
                ],
                "service_ids": ["001", "002", "003", "004"]
            }
        }

@rag_router.post("/api/rag", response_model=RagResponse)
async def get_rag_results(request: RagRequest):
    try:
        # main 함수 호출
        documents_dict, common_ids, vector_only_ids, sql_only_ids = await main(request.query, request.user_id)
        
        # 응답 데이터 준비
        # sources 형식으로 변환
        sources = []
        all_service_ids = []
        
        # common_ids, vector_only_ids, sql_only_ids를 모두 합쳐서 중복 없이 service_ids 생성
        # 타입 안전성을 위해 모든 값을 리스트로 변환
        common_ids_list = list(common_ids) if isinstance(common_ids, set) else common_ids
        vector_only_ids_list = list(vector_only_ids) if isinstance(vector_only_ids, set) else vector_only_ids
        sql_only_ids_list = list(sql_only_ids) if isinstance(sql_only_ids, set) else sql_only_ids
        
        # 이제 모두 리스트이므로 안전하게 합칠 수 있음
        all_service_ids = list(set(common_ids_list + vector_only_ids_list + sql_only_ids_list))
        
        # documents_dict에서 sources 정보 추출
        for doc_key, doc_info in documents_dict.items():
            metadata = doc_info.get("메타데이터", {})
            source = {
                "title": metadata.get("title", "제목 정보 없음"),
                "content": doc_info.get("내용", "내용 없음"),
                "service_id": metadata.get("service_id", "ID 없음"),
                "eligibility": metadata.get("eligibility", "자격 조건 정보 없음"),
                "benefits": metadata.get("benefits", "혜택 정보 없음")
            }
            sources.append(source)
        
        # 디버깅을 위한 로그 추가
        print("결과 생성 완료:")
        print(f"- documents_dict 크기: {len(documents_dict)}")
        print(f"- common_ids 크기: {len(common_ids)}")
        print(f"- vector_only_ids 크기: {len(vector_only_ids)}")
        print(f"- sql_only_ids 크기: {len(sql_only_ids)}")
        print(f"- sources 크기: {len(sources)}")
        
        response_data = {
            "documents": documents_dict,
            "common_ids": common_ids_list,
            "vector_only_ids": vector_only_ids_list,
            "sql_only_ids": sql_only_ids_list,
            "sources": sources,
            "service_ids": all_service_ids
        }
        
        # 응답 데이터 유효성 검사 (디버깅 목적)
        try:
            # 각 필드 타입 확인
            assert isinstance(response_data["documents"], dict), "documents는 딕셔너리여야 합니다"
            assert isinstance(response_data["common_ids"], list), "common_ids는 리스트여야 합니다"
            assert isinstance(response_data["vector_only_ids"], list), "vector_only_ids는 리스트여야 합니다"
            assert isinstance(response_data["sql_only_ids"], list), "sql_only_ids는 리스트여야 합니다"
            assert isinstance(response_data["sources"], list), "sources는 리스트여야 합니다"
            assert isinstance(response_data["service_ids"], list), "service_ids는 리스트여야 합니다"
            
            # 응답 반환
            return response_data
            
        except AssertionError as ae:
            print(f"응답 데이터 타입 검증 실패: {str(ae)}")
            raise HTTPException(status_code=500, detail=f"응답 데이터 형식 오류: {str(ae)}")
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"상세 오류 정보: {error_trace}")
        raise HTTPException(status_code=500, detail=f"RAG 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True) 