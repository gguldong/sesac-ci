from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets
from app.core.db import engine
from app.models.user import Base, chat_message, UserFavoritePolicy # Base에 모든 ORM 모델이 등록되어 있어야 합니다.

# chat_message 테이블만 삭제하고 다시 생성
#chat_message.__table__.drop(engine, checkfirst=True)
#UserFavoritePolicy.__table__.drop(engine, checkfirst=True)
# 등록된 테이블이 없으면 생성
Base.metadata.create_all(engine)

# 라우터 임포트
from app.api import auth, chat, stt, favorites
# 내부 API 라우터 임포트
from app.api import policy_classifier_api

app = FastAPI()

# CORS 설정
origins = ["https://www.jungchat.shop", "http://www.jungchat.shop", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어 (운영 시에는 고정된 secret key 사용 권장)
#secret_key = secrets.token_hex(32)
secret_key = 'P@ssw0rd'
app.add_middleware(
    SessionMiddleware,
    secret_key=secret_key,
    session_cookie="session_id",
    max_age=86400,
    same_site="lax",
)

# API 라우터 등록
app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(stt.router, prefix="/api")
# app.include_router(favorites.router, prefix="/favorites", tags=["favorites"])
# app.include_router(favorites.router, prefix="/api", tags=["favorites"])
app.include_router(favorites.router, prefix="/api/favorites", tags=["favorites"])
# 내부 API 라우터 등록 - /internal 경로로 분리
app.include_router(policy_classifier_api.router, prefix="/internal/policy", tags=["internal"])


