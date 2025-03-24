from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import bcrypt
from app.core.db import get_db
from app.models.user import UserData
from app.schemas.user import UserDataRequest, LoginRequest

router = APIRouter()

@router.post("/submit")
def submit_user_data(user: UserDataRequest, db: Session = Depends(get_db)):
    existing_user = db.query(UserData).filter(UserData.user_id == user.user_id).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    personal_char_str = ",".join(user.personalCharacteristics)
    household_char_str = ",".join(user.householdCharacteristics)
    
    db_user = UserData(
        user_id=user.user_id,
        password=hashed_password.decode('utf-8'),
        username=user.username,
        email=user.email,
        phone=user.phone,
        area=user.area,
        district=user.district,
        birthDate=user.birthDate,
        gender=user.gender,
        incomeRange=user.incomeRange,
        personalCharacteristics=personal_char_str,
        householdCharacteristics=household_char_str
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "데이터 저장 성공", "id": db_user.user_id}

@router.post("/login")
async def login(login_req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    # 사용자 존재 여부 확인
    user = db.query(UserData).filter(UserData.user_id == login_req.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="존재하지 않는 아이디입니다.")
    
    # 비밀번호 검증
    if not bcrypt.checkpw(login_req.password.encode("utf-8"), user.password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

    # 세션에 사용자 ID 저장 (세션 쿠키의 옵션도 확인할 것)
    request.session["user_id"] = user.user_id

    # 클라이언트에 반환할 데이터: 필요한 필드만 포함 (비밀번호 등 제외)
    return {"message": "로그인 성공!", "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "phone": user.phone,
            "birthDate" : user.birthDate
        }}


@router.get("/me")
async def me(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인 상태가 아닙니다.")
    user = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    return {
        "message": f"안녕하세요, {user.username}님!",
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "phone": user.phone
    }

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()  # 세션 삭제
    return {"message": "로그아웃 완료"}
