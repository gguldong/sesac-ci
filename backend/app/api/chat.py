from fastapi import APIRouter, HTTPException, Depends, Query
from app.schemas.chat import ChatMessage, ChatSession, ChatSessionCreate, ChatRequest, ChatResponse
from typing import Dict, List
from uuid import uuid4
from app.services.openai_service import request_gpt
from app.services.openchat_service import request_openchat
from datetime import datetime
from app.core.db import get_db
from app.models.user import chat_session, chat_message, UserData
from sqlalchemy.orm import Session
from app.services.generate_id import generate_unique_message_id
from app.services.policy_classifier import classify_policy_question
from app.services.rag_service import retrieve_relevant_policies

router = APIRouter()

# In-memory storage for demonstration (replace with a database in production)
chat_sessions: Dict[str, List[Dict]] = {}

# Endpoint to create a new chat session
@router.post("/sessions", response_model=ChatSessionCreate)
async def create_session(session_data: ChatSessionCreate, db: Session = Depends(get_db)):

    session_id = str(uuid4()) #중복방지처리 안됨
    chat_sessions[session_id] = []  # initialize an empty message list

        # 이미 동일한 session_id가 존재하는지 확인 (선택 사항)
    existing = db.query(chat_session).filter(chat_session.session_id == session_data.session_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Session already exists")
    
    new_session = chat_session(
        session_id = session_id,
        user_id = session_data.user_id,
        header_message = session_data.header_message
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return ChatSessionCreate(session_id=session_id)

'''
@app.post("/sessions/{session_id}/message") 라는 경로에 있는 {session_id} 부분이
프론트엔드에서 전달한 세션 아이디로 채워지고, 함수의 인자로 session_id: str에 할당됩니다.
'''
# Endpoint to add a message to a session
@router.post("/sessions/{session_id}/message")
async def add_message(session_id: str, message: ChatMessage, db: Session = Depends(get_db)):
    # 모든 세션의 session_id를 조회합니다.
    sessions = db.query(chat_session.session_id).all()
    if not sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    message_dict = message.dict()
    # 메모리 내 저장
    #chat_sessions[session_id].append(message_dict)

        # ORM 인스턴스 생성; message_data['timeStamp']는 이미 datetime 객체임
    new_message = chat_message(
        message_id = generate_unique_message_id(),  # 고유 메시지 ID 생성 함수 (구현 필요)
        session_id = session_id,
        sender = message_dict['sender'],
        message = message_dict['message'],
        timestamp = message_dict['timestamp']  # 프론트엔드에서 전달받은 datetime 값 사용
    )
    
    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    #chat_session.updated_at 업데이트
    db.query(chat_session).filter(chat_session.session_id == session_id).update({
        chat_session.updated_at: datetime.now()
    })
    db.commit()

    return {"status": "success"}

# 특정 세션의 채팅 기록 조회 엔드포인트
@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str, db: Session = Depends(get_db)):
        # 데이터베이스에서 해당 세션을 조회합니다.
    session_obj = db.query(chat_session).filter(chat_session.session_id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 해당 세션의 채팅 메시지를 타임스탬프 순으로 조회합니다.
    messages = (
        db.query(chat_message)
          .filter(chat_message.session_id == session_id)
          .order_by(chat_message.timestamp)
          .all()
    )
        # Pydantic 스키마로 변환하여 반환합니다.
    return ChatSession(
        session_id=session_obj.session_id,
        messages=messages  # ChatSession 스키마의 messages 필드가 채팅 메시지 목록을 받도록 정의되어 있어야 합니다.
    )

# 모든 세션 목록 조회 (사이드바에 표시)
@router.get("/sessions", response_model=List[dict])
async def list_sessions(user_id: str = Query(..., description="User ID"), db: Session = Depends(get_db)):
    # 해당 유저의 session_id와 header_message를 조회합니다. updated_at 오름차순 정렬
    sessions = db.query(chat_session.session_id, chat_session.header_message).filter(chat_session.user_id == user_id).order_by(chat_session.updated_at.desc()).all()
    # sessions는 예: [('session1', 'Hello World'), ('session2', 'Hi there'), ...] 형태이므로 딕셔너리로 변환합니다.
    return [{"sessionId": s[0], "header_message": s[1]} for s in sessions]

# 챗봇 엔드포인트 
@router.post("/model", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest, db: Session = Depends(get_db)):
    # 세션 ID가 제공된 경우 이전 대화 컨텍스트 불러오기
    conversation_context = ""
    
    if chat.session_id:
        # 최근 N개의 메시지만 불러오기 (토큰 제한 고려)
        recent_messages = (
            db.query(chat_message)
            .filter(chat_message.session_id == chat.session_id)
            .order_by(chat_message.timestamp.desc())
            .limit(10)  # 최근 10개 메시지만 사용
            .all()
        )
        
        # 시간 순서대로 정렬
        recent_messages.reverse()
        
        # 컨텍스트 구성
        for msg in recent_messages:
            role = "user111" if msg.sender == "user" else "assistant222"
            conversation_context += f"{role}::: {msg.message}\n"
        
        user_id  = None
        # 선택적: 사용자 정보 추가
        session_obj = db.query(chat_session).filter(chat_session.session_id == chat.session_id).first()
        if session_obj:
            user_obj = db.query(UserData).filter(UserData.user_id == session_obj.user_id).first()
            if user_obj:
                user_id = user_obj.user_id
                # 필요한 사용자 정보만 포함
                user_context = f"사용자 정보: 성별={user_obj.gender}, 지역={user_obj.area}, 특성={user_obj.personalCharacteristics}\n\n"
                conversation_context = user_context + conversation_context
    
    # 통합된 처리 프로세스 적용
    try:
        # 1. 정책 관련 질문인지 분류 - API 호출 대신 직접 함수 호출
        print(f"정책 분류 함수 호출 시작: {chat.message}")
        
        # 내부 API 호출 대신 직접 함수 호출
        is_policy_question = classify_policy_question(chat.message)
            
        print(f"질문 분류 결과: {'정책 관련' if is_policy_question else '일반 질문'}")
        
        # 2. 정책 관련 질문이면 RAG 시스템 사용
        rag_context = ""
        if is_policy_question:
            try:
                # 내부 API 호출 대신 직접 함수 호출
                rag_result = await retrieve_relevant_policies(
                    query=chat.message,
                    user_id=user_id,
                )
                print(f"RAG 결과: {rag_result}")
                print(f"RAG 결과 타입: {type(rag_result)}")
                
                # RAG 결과를 프롬프트에 포함
                if rag_result and isinstance(rag_result, dict):
                    rag_context = "다음은 검색된 관련 정책 정보입니다:\n\n"
                    
                    # sources 필드에서 정책 정보 추출 (상위 15개만)
                    if 'sources' in rag_result and rag_result['sources']:
                        sources = rag_result['sources']
                        # 상위 15개로 제한
                        limited_sources = sources[:15]
                        for index, policy in enumerate(limited_sources):
                            rag_context += f"### {index + 1}. {policy.get('title', '제목 없음')}\n"
                            rag_context += f"{policy.get('content', '내용 없음')}\n"
                            rag_context += f"자격 조건: {policy.get('eligibility', '자격 조건 정보 없음')}\n"
                            rag_context += f"혜택: {policy.get('benefits', '혜택 정보 없음')}\n\n"
                    
                    # 추가 컨텍스트 정보 설정 (vector_only_ids, common_ids, sql_only_ids 제거)
                    rag_context += f"\n이 정보를 바탕으로 사용자 질문에 답변해 주세요.\n"
            except Exception as e:
                print(f"정책 검색 함수 호출 중 오류: {str(e)}")

        # 3. 선택된 모델로 응답 생성
        if chat.model == "openchat":
            response = await request_openchat(
                message=chat.message,
                conversation_history=conversation_context,
                rag_context=rag_context
            )
        else:  # 기본 모델은 GPT
            response = await request_gpt(
                message=chat.message,
                conversation_history=conversation_context,
                rag_context=rag_context
            )
    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
        # 오류 발생 시 사용자에게 일반적인 에러 메시지 반환
        response = "죄송합니다. 시스템에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
    
    return ChatResponse(response=response)

# init_message = None
# @router.get("/model/first", out_message=InitMessage)
# async def first_message(InitMessage: InitMessage):
#     if InitMessage.init_message:
#         init_message = InitMessage.init_message    
#     elif InitMessage.init_message == None:
#         return init_message