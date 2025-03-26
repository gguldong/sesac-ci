from sqlalchemy import Column, String, Date, ForeignKey, Text, DateTime, JSON, Integer
from app.core.db import Base
from datetime import datetime

class UserData(Base):
    __tablename__ = "user"
    user_id = Column(String(100), primary_key=True, index=True)
    password = Column(String(255))
    username = Column(String(30))
    email = Column(String(100))
    phone = Column(String(20))
    area = Column(String(100))
    district = Column(String(100))
    birthDate = Column(Date)
    gender = Column(String(10))
    incomeRange = Column(String(100))
    personalCharacteristics = Column(String(255))
    householdCharacteristics = Column(String(255))

class UserFavoritePolicy(Base):
    __tablename__ = "user_favorite_policy"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("user.user_id"), nullable=False, index=True)
    policy_id = Column(String(255), nullable=False)  # 정책 ID (제목 또는 고유 식별자)
    created_at = Column(DateTime, default=datetime.now)
    
    # 복합 유니크 제약조건 - 같은 사용자가 같은 정책을 중복해서 추가할 수 없음
    __table_args__ = (
        {"sqlite_autoincrement": True},  # SQLite에서 autoincrement를 활성화
    )

class chat_session(Base):
    __tablename__ = "chat_session"
    session_id = Column(String(100), primary_key=True, index=True)  # 기본키 추가
    user_id = Column(String(100), ForeignKey("user.user_id"), nullable=False)  # FK 설정
    header_message = Column(String(255), default="Default Header")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class chat_message(Base):
    __tablename__ = "chat_message"
    message_id = Column(String(100), primary_key=True, index=True)  # 기본키 추가
    session_id = Column(String(100), ForeignKey("chat_session.session_id"), nullable=False)
    sender = Column(String(20))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)  # 기본값을 현재 UTC 시간으로 지정
    policies = Column(JSON, nullable=True)  # 정책 정보를 JSON 형태로 저장
