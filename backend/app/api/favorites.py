from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re

from app.core.db import get_db
from app.models.user import UserFavoritePolicy
from app.services.policy_api import fetch_policy_details, format_policy_for_frontend

load_dotenv()

router = APIRouter()

# 신청기한 정보를 파싱하여 타입을 결정하는 함수
def parse_application_period(period_text):
    if not period_text:
        return "기타"
        
    # 상시 신청 키워드 확인
    permanent_keywords = ["상시", "연중", "수시", "언제든지", "제한없음", "언제나"]
    for keyword in permanent_keywords:
        if keyword in period_text:
            return "상시"
    
    # 정기신청/반기신청 키워드 확인
    if "정기" in period_text or "반기" in period_text:
        return "정기"
            
    # 날짜 범위 확인 (다양한 형식)
    date_range_pattern1 = r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*[~-]\s*\d{1,2}월\s*\d{1,2}일"
    date_range_pattern2 = r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*[~-]\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일"
    date_range_pattern3 = r"\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*[~-]\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    date_range_pattern4 = r"\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*[~-]\s*\d{1,2}[./-]\d{1,2}"
    date_range_pattern5 = r"\d{1,2}\.\d{1,2}\.?~\d{1,2}\.\d{1,2}\.?"  # 5.1.~5.31. 형식
    
    if (re.search(date_range_pattern1, period_text) or 
        re.search(date_range_pattern2, period_text) or 
        re.search(date_range_pattern3, period_text) or 
        re.search(date_range_pattern4, period_text) or
        re.search(date_range_pattern5, period_text) or
        "~" in period_text):
        return "정기"
    
    # 규정 참조 키워드 확인
    regulation_keywords = ["규정에 따름", "규정에따름", "규정참조", "별도공지"]
    for keyword in regulation_keywords:
        if keyword in period_text:
            return "규정참조"
        
    # 조건부 신청 키워드 확인
    conditional_keywords = ["조건", "충족", "해당자", "요건", "일정기준", "자격", "기준", "접수후", "접수 후"]
    for keyword in conditional_keywords:
        if keyword in period_text:
            return "조건부"
            
    # 그 외에는 기타로 처리
    return "기타"

# 사용자별 즐겨찾기 목록 조회
@router.get("/{user_id}", response_model=List[dict])
async def get_user_favorites(user_id: str, db: Session = Depends(get_db)):
    """
    사용자의 즐겨찾기 정책 목록을 조회합니다.
    """
    favorites = db.query(UserFavoritePolicy).filter(UserFavoritePolicy.user_id == user_id).all()
    
    # 반환 형식 변환
    result = []
    for fav in favorites:
        result.append({
            "id": fav.id,
            "user_id": fav.user_id,
            "policy_id": fav.policy_id,
            "created_at": fav.created_at.isoformat() if fav.created_at else None
        })
    
    return result

# 사용자별 즐겨찾기 정책 상세 정보 조회 (캘린더용)
@router.get("/{user_id}/calendar", response_model=List[dict])
async def get_user_favorites_calendar(user_id: str, db: Session = Depends(get_db)):
    """
    사용자의 즐겨찾기 정책을 정부24 API를 통해 상세 정보와 함께 조회합니다.
    """
    # 사용자의 즐겨찾기 목록 조회
    favorites = db.query(UserFavoritePolicy).filter(UserFavoritePolicy.user_id == user_id).all()
    
    if not favorites:
        return []
    
    # 정책 상세 정보 조회 결과
    detailed_policies = []
    
    # 각 즐겨찾기 항목에 대해 정책 상세 정보 조회
    for fav in favorites:
        try:
            # 정책 ID 사용
            policy_id = fav.policy_id
            print(f"즐겨찾기 id: {policy_id}")
            # 정책 ID가 API에서 사용할 수 있는 형태인지 확인
            if not policy_id or policy_id.strip() == "":
                # 정책 ID가 없는 경우 샘플 데이터 구성
                sample_policy = {
                    "id": "sample_" + str(datetime.now().timestamp()),  # 고유 ID 생성
                    "service_id": "",  # 빈 서비스 ID
                    "title": "알 수 없는 정책",
                    "content": "정책 ID가 없습니다",
                    "type": "여유있는 정책",
                    "startDate": datetime.now().strftime("%Y-%m-%d"),
                    "endDate": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d"),
                    "application_period": "정보 없음",
                    "application_type": "기타"
                }
                
                detailed_policies.append(sample_policy)
                continue
                
            # 정부24 API 호출 - policy_id를 그대로 service_id로 사용
            policy_detail = await fetch_policy_details(policy_id)
            
            if policy_detail:
                # API 결과를 프론트엔드 형식으로 변환 - format_policy_for_frontend 함수가 id와 field 매핑 처리
                formatted_policy = format_policy_for_frontend(policy_detail)
                if formatted_policy:
                    # API 모듈에서 이미 application_type이 설정되어 있으면 그대로 사용
                    # 그렇지 않은 경우 추가 파싱 수행
                    if not formatted_policy.get("application_type"):
                        formatted_policy["application_type"] = parse_application_period(formatted_policy.get("application_period", ""))
                    
                    detailed_policies.append(formatted_policy)
                    print(f"정책 상세 정보 조회 성공: {policy_id}")
                else:
                    print(f"정책 상세 정보 형식 변환 실패: {policy_id}")
                    # 기본 정책 데이터 추가 (이게 보이면 API 연동은 됐지만 형식 변환에 문제가 있다는 의미)
                    policy_data = {
                        "id": policy_id,  # 원본 정책 ID 유지
                        "service_id": policy_id,  # 서비스 ID도 동일하게 설정
                        "title": f"정책 정보 없음 (ID: {policy_id})",
                        "content": "정책 정보를 불러올 수 없습니다.",
                        "type": "여유있는 정책",
                        "application_period": "정보 없음",
                        "application_type": "기타"
                    }
                    
                    detailed_policies.append(policy_data)
            else:
                print(f"정책 상세 정보 조회 실패: {policy_id}")
                # 기본 정책 데이터 추가 (이게 보이면 API 연동에 문제가 있다는 의미)
                policy_data = {
                    "id": policy_id,  # 원본 정책 ID 유지
                    "service_id": policy_id,  # 서비스 ID도 동일하게 설정
                    "title": f"정책 정보 없음 (ID: {policy_id})", 
                    "content": "정책 정보를 불러올 수 없습니다.",
                    "type": "여유있는 정책",
                    "application_period": "정보 없음",
                    "application_type": "기타"
                }
                
                detailed_policies.append(policy_data)
        except Exception as e:
            print(f"정책 상세 정보 조회 중 오류 발생: {str(e)}")
            # 오류 발생 시에도 기본 정책 데이터 추가
            policy_data = {
                "id": fav.policy_id,  # 원본 정책 ID 유지
                "service_id": fav.policy_id,  # 서비스 ID도 동일하게 설정
                "title": f"정책 정보 없음 (ID: {fav.policy_id})",
                "content": "오류로 인해 정책 정보를 불러올 수 없습니다.",
                "type": "여유있는 정책",
                "application_period": "정보 없음",
                "application_type": "기타"
            }
            
            detailed_policies.append(policy_data)
    
    return detailed_policies

# 즐겨찾기 추가
@router.post("", status_code=status.HTTP_201_CREATED)
async def add_favorite(favorite: dict, db: Session = Depends(get_db)):
    """
    정책을 즐겨찾기에 추가합니다.
    """
    # 필수 필드 검증
    if not favorite.get("user_id") or not favorite.get("policy_id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="사용자 ID와 정책 ID는 필수 입력값입니다."
        )
    
    # 이미 존재하는지 확인
    existing = db.query(UserFavoritePolicy).filter(
        UserFavoritePolicy.user_id == favorite["user_id"],
        UserFavoritePolicy.policy_id == favorite["policy_id"]
    ).first()
    
    if existing:
        return {"message": "이미 즐겨찾기에 추가된 정책입니다."}
    
    # 새 즐겨찾기 항목 생성
    try:
        new_favorite = UserFavoritePolicy(
            user_id=favorite["user_id"],
            policy_id=favorite["policy_id"],
            created_at=datetime.now()
        )
        
        db.add(new_favorite)
        db.commit()
        db.refresh(new_favorite)
        
        return {
            "id": new_favorite.id,
            "user_id": new_favorite.user_id,
            "policy_id": new_favorite.policy_id,
            "created_at": new_favorite.created_at.isoformat()
        }
    
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="즐겨찾기 추가 중 오류가 발생했습니다."
        )

# 즐겨찾기 삭제
@router.delete("/{user_id}/{policy_id}", status_code=status.HTTP_200_OK)
async def remove_favorite(user_id: str, policy_id: str, db: Session = Depends(get_db)):
    """
    즐겨찾기에서 정책을 제거합니다.
    """
    favorite = db.query(UserFavoritePolicy).filter(
        UserFavoritePolicy.user_id == user_id,
        UserFavoritePolicy.policy_id == policy_id
    ).first()
    
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="해당 즐겨찾기 항목을 찾을 수 없습니다."
        )
    
    try:
        db.delete(favorite)
        db.commit()
        return {"message": "즐겨찾기에서 정책이 제거되었습니다."}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"즐겨찾기 제거 중 오류가 발생했습니다: {str(e)}"
        )

# 특정 사용자의 특정 정책 즐겨찾기 여부 확인
@router.get("/{user_id}/{policy_id}", response_model=dict)
async def check_favorite(user_id: str, policy_id: str, db: Session = Depends(get_db)):
    """
    특정 정책이 사용자의 즐겨찾기에 있는지 확인합니다.
    """
    favorite = db.query(UserFavoritePolicy).filter(
        UserFavoritePolicy.user_id == user_id,
        UserFavoritePolicy.policy_id == policy_id
    ).first()
    
    if favorite:
        return {
            "is_favorite": True,
            "id": favorite.id,
            "user_id": favorite.user_id,
            "policy_id": favorite.policy_id,
            "created_at": favorite.created_at.isoformat() if favorite.created_at else None
        }
    else:
        return {"is_favorite": False}
