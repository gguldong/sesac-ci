from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional, Any

class ChatMessage(BaseModel):
    timestamp: datetime # ISO 8601 형식의 문자열 → datetime 객체로 변환됨
    sender: str
    message: str
    message_id: Optional[str] = None
    policies: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True  # pydantic v2에서는 orm_mode 대신 사용
        populate_by_name = True  # 내부 필드명("text")도 허용됨


class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class ChatSessionCreate(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    header_message: str | None = None

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # 선택적 세션 ID 추가
    model: str = "openchat"  # 모델 선택 


class ChatResponse(BaseModel):
    response: str
    policies: Optional[List[Dict[str, Any]]] = None


# class InitMessage(BaseModel):
#     init_message: str | None = None