# from fastapi import APIRouter, HTTPException, Depends, Query
# from app.schemas.chat import ChatMessage, ChatSession, ChatSessionCreate, ChatRequest, ChatResponse
# from typing import Dict, List, Optional
# from uuid import uuid4
# from app.services.openai_service import request_gpt
# from datetime import datetime
# from app.core.db import get_db
# from app.models.user import chat_session, chat_message, UserData
# from sqlalchemy.orm import Session
# from app.services.generate_id import generate_unique_message_id
# import json
# import time
# from redis import Redis
# import logging

# # # 로깅 설정
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # Redis 연결
# try:
#     redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
#     redis_client.ping()  # 연결 테스트
#     logger.info("Redis 연결 성공")
# except Exception as e:
#     logger.error(f"Redis 연결 실패: {str(e)}")
#     redis_client = None

# router = APIRouter()

# # Redis 키 상수
# SESSION_USER_PREFIX = "session_user:"
# CONTEXT_PREFIX = "context:"
# SESSION_PREFIX = "session:"
# RATE_LIMIT_PREFIX = "rate_limit:"

# # 유틸리티 함수
# def get_from_redis(key):
#     """안전하게 Redis에서 데이터 조회"""
#     if redis_client is None:
#         return None
#     try:
#         return redis_client.get(key)
#     except Exception as e:
#         logger.error(f"Redis 조회 실패 ({key}): {str(e)}")
#         return None

# def set_to_redis(key, value, expire=None):
#     """안전하게 Redis에 데이터 저장"""
#     if redis_client is None:
#         return False
#     try:
#         if expire:
#             redis_client.setex(key, expire, value)
#         else:
#             redis_client.set(key, value)
#         return True
#     except Exception as e:
#         logger.error(f"Redis 저장 실패 ({key}): {str(e)}")
#         return False

# def delete_from_redis(key):
#     """안전하게 Redis 키 삭제"""
#     if redis_client is None:
#         return False
#     try:
#         redis_client.delete(key)
#         return True
#     except Exception as e:
#         logger.error(f"Redis 삭제 실패 ({key}): {str(e)}")
#         return False

# def rate_limit(key, max_requests, time_window):
#     """요청 속도 제한 구현"""
#     if redis_client is None:
#         return True  # Redis 없으면 항상 통과
    
#     try:
#         current_time = int(time.time())
#         requests = redis_client.zrangebyscore(key, current_time - time_window, current_time)
        
#         if len(requests) >= max_requests:
#             return False
            
#         redis_client.zadd(key, {str(current_time): current_time})
#         redis_client.zremrangebyscore(key, 0, current_time - time_window)
#         return True
#     except Exception as e:
#         logger.error(f"요율 제한 오류 ({key}): {str(e)}")
#         return True  # 오류 시 제한 없이 통과

# @router.post("/sessions", response_model=ChatSessionCreate)
# async def create_session(session_data: ChatSessionCreate, db: Session = Depends(get_db)):
#     # 요율 제한 
#     key = f"{RATE_LIMIT_PREFIX}{session_data.user_id}"
#     if not rate_limit(key, max_requests=10, time_window=60):
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     # 새 세션 생성
#     session_id = str(uuid4())
    
#     # DB에 세션 저장
#     new_session = chat_session(
#         session_id=session_id,
#         user_id=session_data.user_id,
#         header_message=session_data.header_message,
#         created_at=datetime.now(),
#         updated_at=datetime.now()
#     )
#     db.add(new_session)
#     db.commit()
#     db.refresh(new_session)
    
#     # Redis에 세션-사용자 매핑 저장
#     set_to_redis(f"{SESSION_USER_PREFIX}{session_id}", session_data.user_id)
    
#     # 사용자 정보 캐싱
#     user_obj = db.query(UserData).filter(UserData.user_id == session_data.user_id).first()
#     if user_obj:
#         initial_context = f"사용자 정보: 성별={user_obj.gender}, 지역={user_obj.area}, 특성={user_obj.personalCharacteristics}\n\n"
#         set_to_redis(f"{CONTEXT_PREFIX}{session_id}", initial_context, expire=3600)
    
#     return ChatSessionCreate(session_id=session_id)

# @router.post("/sessions/{session_id}/message")
# async def add_message(session_id: str, message: ChatMessage, db: Session = Depends(get_db)):
#     # 사용자 ID 찾기
#     user_id = get_from_redis(f"{SESSION_USER_PREFIX}{session_id}")
#     if not user_id:
#         # Redis에 없으면 DB에서 조회
#         session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
#         if not session_obj:
#             raise HTTPException(status_code=404, detail="Session not found")
#         user_id = session_obj.user_id
#         set_to_redis(f"{SESSION_USER_PREFIX}{session_id}", user_id)
    
#     # 요율 제한
#     if not rate_limit(f"{RATE_LIMIT_PREFIX}{user_id}", 10, 60):
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     # 메시지 저장
#     message_dict = message.dict()
#     timestamp = message_dict['timestamp']
#     if isinstance(timestamp, str):
#         timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
#     new_message = chat_message(
#         message_id=generate_unique_message_id(),
#         session_id=session_id,
#         sender=message_dict['sender'],
#         message=message_dict['message'],
#         timestamp=timestamp
#     )
    
#     db.add(new_message)
#     db.commit()
    
#     # 세션 갱신
#     db.query(chat_session).filter(chat_session.session_id == session_id).update({
#         chat_session.updated_at: datetime.now()
#     })
#     db.commit()

#     # Redis 캐시 무효화
#     delete_from_redis(f"{SESSION_PREFIX}{session_id}")
    
#     # 컨텍스트 업데이트
#     context = get_from_redis(f"{CONTEXT_PREFIX}{session_id}") or ""
#     role = "user" if message_dict['sender'] == "user" else "assistant"
#     updated_context = context + f"{role}::: {message_dict['message']}\n"
#     set_to_redis(f"{CONTEXT_PREFIX}{session_id}", updated_context, expire=3600)
    
#     return {"status": "success"}

# @router.get("/sessions/{session_id}", response_model=ChatSession)
# async def get_session(session_id: str, db: Session = Depends(get_db)):
#     # 사용자 ID 찾기
#     user_id = get_from_redis(f"{SESSION_USER_PREFIX}{session_id}")
#     if not user_id:
#         session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
#         if not session_obj:
#             raise HTTPException(status_code=404, detail="Session not found")
#         user_id = session_obj.user_id
#         set_to_redis(f"{SESSION_USER_PREFIX}{session_id}", user_id)
    
#     # 요율 제한
#     if not rate_limit(f"{RATE_LIMIT_PREFIX}{user_id}", 10, 60):
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     # Redis 캐시 확인
#     cached_session = get_from_redis(f"{SESSION_PREFIX}{session_id}")
#     if cached_session:
#         try:
#             return ChatSession(**json.loads(cached_session))
#         except Exception as e:
#             logger.error(f"캐시 데이터 파싱 실패: {str(e)}")
#             # 오류 시 DB에서 로드
    
#     # DB에서 세션 로드
#     session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
#     if not session_obj:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     messages = (
#         db.query(chat_message)
#         .filter(chat_message.session_id == session_id)
#         .order_by(chat_message.timestamp)
#         .all()
#     )
    
#     result = ChatSession(session_id=session_obj.session_id, messages=messages)
    
#     # Redis에 캐싱
#     set_to_redis(f"{SESSION_PREFIX}{session_id}", json.dumps(result.dict()), expire=3600)
    
#     return result

# @router.get("/sessions", response_model=List[dict])
# async def list_sessions(user_id: str = Query(..., description="User ID"), db: Session = Depends(get_db)):
#     # 요율 제한
#     if not rate_limit(f"{RATE_LIMIT_PREFIX}{user_id}", 10, 60):
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     # 세션 목록 조회
#     sessions = db.query(chat_session.session_id, chat_session.header_message).filter(
#         chat_session.user_id == user_id
#     ).order_by(chat_session.updated_at.desc()).all()
    
#     return [{"sessionId": s[0], "header_message": s[1]} for s in sessions]

# @router.post("/model")
# async def chat_endpoint(chat: ChatRequest, db: Session = Depends(get_db)):
#     session_id = chat.session_id
#     if not session_id:
#         raise HTTPException(status_code=400, detail="Session ID is required")
    
#     # 사용자 ID 찾기
#     user_id = get_from_redis(f"{SESSION_USER_PREFIX}{session_id}")
#     if not user_id:
#         session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
#         if not session_obj:
#             raise HTTPException(status_code=404, detail="Session not found")
#         user_id = session_obj.user_id
#         set_to_redis(f"{SESSION_USER_PREFIX}{session_id}", user_id)
    
#     # 요율 제한
#     if not rate_limit(f"{RATE_LIMIT_PREFIX}{user_id}", 10, 60):
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     # 대화 컨텍스트 로드
#     conversation_context = get_from_redis(f"{CONTEXT_PREFIX}{session_id}")
#     if not conversation_context:
#         # Redis에 없으면 DB에서 로드
#         recent_messages = (
#             db.query(chat_message)
#             .filter(chat_message.session_id == session_id)
#             .order_by(chat_message.timestamp.desc())
#             .limit(10)
#             .all()
#         )
        
#         recent_messages.reverse()
#         conversation_context = ""
        
#         for msg in recent_messages:
#             role = "user" if msg.sender == "user" else "assistant"
#             conversation_context += f"{role}::: {msg.message}\n"
        
#         # 사용자 정보 추가
#         session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
#         if session_obj:
#             user_obj = db.query(UserData).filter(UserData.user_id == session_obj.user_id).first()
#             if user_obj:
#                 user_context = f"사용자 정보: 성별={user_obj.gender}, 지역={user_obj.area}, 특성={user_obj.personalCharacteristics}\n\n"
#                 conversation_context = user_context + conversation_context
        
#         # Redis에 캐싱
#         set_to_redis(f"{CONTEXT_PREFIX}{session_id}", conversation_context, expire=3600)
    
#     # 선택된 모델에 따라 응답 생성
#     gpt_response = ""
#     if chat.model == "rag":
#         try:
#             # RAG 시스템 사용
#             from app.services.rag_service import retrieve_relevant_policies
            
#             # RAG 시스템 호출 (세션 ID를 통해 사용자 정보 사용)
#             policies = retrieve_relevant_policies(
#                 query=chat.message,
#                 session_id=session_id
#             )
            
#             # 정책 정보가 있으면 응답 생성
#             if policies and len(policies) > 0:
#                 gpt_response = format_policy_response(policies)
#             else:
#                 # 정책 정보가 없으면 기본 모델 사용
#                 gpt_response = request_gpt(
#                     message=chat.message,
#                     conversation_history=conversation_context
#                 )
#         except Exception as e:
#             print(f"RAG 처리 중 오류 발생: {str(e)}")
#             # 오류 발생 시 기본 모델 사용
#             gpt_response = request_gpt(
#                 message=chat.message,
#                 conversation_history=conversation_context
#             )
#     else:
#         # OpenChat 또는 기본 모델 사용
#         gpt_response = request_gpt(
#             message=chat.message,
#             conversation_history=conversation_context,
#             model=chat.model
#         )
    
#     # 현재 시간
#     current_time = datetime.now()
    
#     # 메시지 저장 (DB)
#     user_message = chat_message(
#         message_id=generate_unique_message_id(),
#         session_id=session_id,
#         sender="user",
#         message=chat.message,
#         timestamp=current_time
#     )
    
#     bot_message = chat_message(
#         message_id=generate_unique_message_id(),
#         session_id=session_id,
#         sender="bot",
#         message=gpt_response,
#         timestamp=current_time
#     )
    
#     db.add(user_message)
#     db.add(bot_message)
#     db.commit()
    
#     # 세션 업데이트
#     db.query(chat_session).filter(chat_session.session_id == session_id).update({
#         chat_session.updated_at: current_time
#     })
#     db.commit()
    
#     # Redis 캐시 무효화/업데이트
#     delete_from_redis(f"{SESSION_PREFIX}{session_id}")
    
#     # 대화 컨텍스트 업데이트
#     updated_context = conversation_context + f"user::: {chat.message}\n" + f"assistant::: {gpt_response}\n"
#     set_to_redis(f"{CONTEXT_PREFIX}{session_id}", updated_context, expire=3600)
    
#     # 프론트엔드에 반환
#     return {
#         "response": gpt_response,
#         "messages": [
#             {
#                 "message_id": str(user_message.message_id),
#                 "timestamp": current_time.isoformat(),
#                 "sender": "user",
#                 "message": chat.message
#             },
#             {
#                 "message_id": str(bot_message.message_id),
#                 "timestamp": current_time.isoformat(),
#                 "sender": "bot",
#                 "message": gpt_response
#             }
#         ]
#     }

# # 정책 정보를 응답 형식으로 변환하는 함수
# def format_policy_response(policies):
#     if not policies or len(policies) == 0:
#         return "죄송합니다. 관련 정책 정보를 찾을 수 없습니다."
    
#     response = "다음과 같은 관련 정책 정보를 찾았습니다:\n\n"
    
#     for index, policy in enumerate(policies):
#         response += f"### {index + 1}. {policy.get('title', '제목 없음')}\n\n"
#         response += f"{policy.get('content', '내용 없음')}\n\n"
#         response += f"**자격 조건**: {policy.get('eligibility', '자격 조건 정보 없음')}\n\n"
#         response += f"**혜택**: {policy.get('benefits', '혜택 정보 없음')}\n\n"
#         response += "---\n\n"
    
#     response += "더 자세한 정보가 필요하시면 추가로 질문해 주세요."
#     return response

# # Redis 상태 확인 엔드포인트
# @router.get("/health/redis")
# async def check_redis_health():
#     if redis_client is None:
#         return {"status": "disconnected", "error": "Redis 연결 없음"}
    
#     try:
#         redis_client.ping()
#         return {
#             "status": "healthy",
#             "active_sessions": len(redis_client.keys(f"{SESSION_PREFIX}*")),
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }