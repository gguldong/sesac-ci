import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json


load_dotenv()

# 정부24 API를 호출하여 서비스 상세 정보를 가져오는 함수
async def fetch_policy_details(service_id: str):
    """
    정부24 API를 호출하여 특정 서비스 ID의 상세 정보를 가져옵니다.
    
    Args:
        service_id: 정책 서비스 ID (문자열)
        
    Returns:
        dict: 정책 상세 정보
    """
    try:
        # API 키 가져오기
        api_key = os.getenv("POLICY_API_KEY")
        if not api_key:
            print("API 키가 설정되지 않았습니다.")
            return None
        
        # API 엔드포인트 URL 구성
        url = f"https://api.odcloud.kr/api/gov24/v3/serviceDetail"
        params = {
            "page": 1,
            "perPage": 1,
            f"cond[서비스ID::EQ]": service_id,
            "serviceKey": api_key
        }
        
        print(f"정책 API 호출: {url}, 서비스 ID: {service_id}")
        
        # API 요청 보내기
        response = requests.get(url, params=params)
        
        # 응답 상태 확인
        response.raise_for_status()
        
        # JSON 응답 파싱
        data = response.json()
        
        print(f"API 응답 코드: {response.status_code}")
        
        # 응답 형태 확인
        # {
        #   "currentCount": 1,
        #   "data": [ { 정책 정보 } ],
        #   "matchCount": 1,
        #   "page": 1,
        #   "perPage": 2,
        #   "totalCount": 10242
        # }
        
        # 결과 반환 (data 배열의 첫 번째 항목)
        if data and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0]
        else:
            print(f"서비스 ID {service_id}에 대한 정책 정보를 찾을 수 없습니다. 응답: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {str(e)}")
        return None
    except Exception as e:
        print(f"정책 정보 조회 중 예상치 못한 오류 발생: {str(e)}")
        return None

# 정책 정보를 프론트엔드에 적합한 형식으로 변환하는 함수
def format_policy_for_frontend(policy_data):
    """
    API에서 받은 정책 정보를 프론트엔드에 적합한 형식으로 변환합니다.
    
    Args:
        policy_data: API에서 반환된 원본 정책 데이터
        
    Returns:
        dict: 프론트엔드에 전달할 형식의 정책 데이터
    """
    if not policy_data:
        return None
    
    try:
        # 날짜 정보 추출 (신청기한에서 일자 정보 추출)
        application_period = policy_data.get("신청기한", "")
        start_date = None
        end_date = None
        
        # 신청기한에서 날짜 정보 파싱
        if not application_period or application_period == "":
            # 정보가 없는 경우 날짜 정보 없음
            start_date = None
            end_date = None
        elif "상시" in application_period or "수시" in application_period or "연중" in application_period:
            # 상시 신청인 경우 현재 년도의 전체 기간으로 설정
            year = datetime.now().year
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
        elif "규정에 따름" in application_period or "규정에따름" in application_period or "규정참조" in application_period:
            # 규정 참조인 경우는 날짜 정보 없음
            start_date = None
            end_date = None
        elif "접수후" in application_period or "접수 후" in application_period or "조건" in application_period or "충족" in application_period:
            # 조건부인 경우 날짜 정보 없음으로 처리 (추측하지 않음)
            start_date = None
            end_date = None
        elif "정기신청" in application_period or "반기신청" in application_period:
            # 정기/반기 신청 형태 파악
            try:
                import re
                
                # 정기신청 기간 패턴 찾기
                regular_period_pattern = r"정기신청\s*:\s*(\d{1,2})\.(\d{1,2})\.?~(\d{1,2})\.(\d{1,2})\.?"
                regular_match = re.search(regular_period_pattern, application_period)
                
                if regular_match:
                    # 현재 날짜와 가장 가까운 정기신청 기간 선택
                    start_month = int(regular_match.group(1))
                    start_day = int(regular_match.group(2))
                    end_month = int(regular_match.group(3))
                    end_day = int(regular_match.group(4))
                    
                    # 년도 정보 추가
                    year = datetime.now().year
                    
                    # 같은 월인 경우 (예: 5.1.~5.31.)
                    if start_month == end_month:
                        start_date = f"{year}-{start_month:02d}-{start_day:02d}"
                        end_date = f"{year}-{end_month:02d}-{end_day:02d}"
                    else:
                        # 다른 월인 경우
                        start_date = f"{year}-{start_month:02d}-{start_day:02d}"
                        end_date = f"{year}-{end_month:02d}-{end_day:02d}"
                else:
                    # 반기신청 기간 패턴 찾기 (상반기/하반기)
                    half_year_pattern = r"반기신청[^:]*상반기[^:]*:\s*(\d{1,2})\.(\d{1,2})\.?~(\d{1,2})\.(\d{1,2})\.?[^:]*하반기[^:]*:\s*(\d{1,2})\.(\d{1,2})\.?~(\d{1,2})\.(\d{1,2})\.?"
                    half_year_match = re.search(half_year_pattern, application_period)
                    
                    if half_year_match:
                        # 현재 날짜 기준으로 상반기/하반기 중 가까운 기간 선택
                        current_month = datetime.now().month
                        year = datetime.now().year
                        
                        # 상반기 정보
                        h1_start_month = int(half_year_match.group(1))
                        h1_start_day = int(half_year_match.group(2))
                        h1_end_month = int(half_year_match.group(3))
                        h1_end_day = int(half_year_match.group(4))
                        
                        # 하반기 정보
                        h2_start_month = int(half_year_match.group(5))
                        h2_start_day = int(half_year_match.group(6))
                        h2_end_month = int(half_year_match.group(7))
                        h2_end_day = int(half_year_match.group(8))
                        
                        # 상/하반기 중 현재 월에 가까운 쪽 선택
                        if abs(current_month - h1_start_month) <= abs(current_month - h2_start_month):
                            start_date = f"{year}-{h1_start_month:02d}-{h1_start_day:02d}"
                            end_date = f"{year}-{h1_end_month:02d}-{h1_end_day:02d}"
                        else:
                            start_date = f"{year}-{h2_start_month:02d}-{h2_start_day:02d}"
                            end_date = f"{year}-{h2_end_month:02d}-{h2_end_day:02d}"
                    else:
                        # 단순 숫자 추출 방식 시도
                        number_pattern = r"(\d{1,2})\.(\d{1,2})\.?~(\d{1,2})\.(\d{1,2})\.?"
                        number_match = re.search(number_pattern, application_period)
                        
                        if number_match:
                            year = datetime.now().year
                            start_month = int(number_match.group(1))
                            start_day = int(number_match.group(2))
                            end_month = int(number_match.group(3))
                            end_day = int(number_match.group(4))
                            
                            start_date = f"{year}-{start_month:02d}-{start_day:02d}"
                            end_date = f"{year}-{end_month:02d}-{end_day:02d}"
                    
                    # 연도가 명시된 날짜 패턴 찾기
                    year_pattern = r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*[~-]\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일"
                    year_match = re.search(year_pattern, application_period)
                    
                    if year_match:
                        start_year = int(year_match.group(1))
                        start_month = int(year_match.group(2))
                        start_day = int(year_match.group(3))
                        
                        # 종료 연도가 있으면 사용, 없으면 시작 연도와 동일하게
                        end_year = int(year_match.group(4)) if year_match.group(4) else start_year
                        end_month = int(year_match.group(5))
                        end_day = int(year_match.group(6))
                        
                        start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"
                        end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
            except Exception as e:
                print(f"정기/반기 신청 날짜 파싱 오류: {str(e)}")
                # 파싱 실패 시 날짜 정보 없음
                start_date = None
                end_date = None
        elif "~" in application_period:
            # 기존 파싱 로직 - "~"로 구분된 날짜
            try:
                import re
                period_parts = application_period.split("~")
                # 간단한 예시: "3월 10일 ~ 3월 20일" 형태 파싱
                start_info = period_parts[0].strip()
                end_info = period_parts[1].strip()
                
                # 연도가 포함된 날짜 패턴 확인
                year_pattern = r"(\d{4})년"
                year_match_start = re.search(year_pattern, start_info)
                year_match_end = re.search(year_pattern, end_info)
                
                year = datetime.now().year
                start_year = year
                end_year = year
                
                if year_match_start:
                    start_year = int(year_match_start.group(1))
                if year_match_end:
                    end_year = int(year_match_end.group(1))
                
                # 월 정보 추출
                current_month = datetime.now().month
                start_month = current_month
                end_month = current_month
                
                if "월" in start_info:
                    month_str = start_info.split("월")[0]
                    # 연도 제거 처리
                    if "년" in month_str:
                        month_str = month_str.split("년")[-1].strip()
                    if month_str.isdigit():
                        start_month = int(month_str)
                
                if "월" in end_info:
                    month_str = end_info.split("월")[0]
                    # 연도 제거 처리
                    if "년" in month_str:
                        month_str = month_str.split("년")[-1].strip()
                    if month_str.isdigit():
                        end_month = int(month_str)
                
                # 일 정보 추출
                start_day = 1
                end_day = get_last_day_of_month(end_month, end_year)
                
                if "일" in start_info:
                    day_str = start_info.split("일")[0].split("월")[-1].strip()
                    if day_str.isdigit():
                        start_day = int(day_str)
                
                if "일" in end_info:
                    day_str = end_info.split("일")[0].split("월")[-1].strip()
                    if day_str.isdigit():
                        end_day = int(day_str)
                
                # 정확한 날짜 정보 생성
                start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"
                end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
                
            except Exception as e:
                print(f"일반 날짜 형식 파싱 오류: {str(e)}")
                # 파싱 실패 시 날짜 정보 없음
                start_date = None
                end_date = None
        
        # ISO 포맷 날짜 패턴 확인 (YYYY-MM-DD)
        if not start_date or not end_date:
            try:
                import re
                iso_pattern = r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*[~-]\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})"
                iso_match = re.search(iso_pattern, application_period)
                
                if iso_match:
                    start_year = int(iso_match.group(1))
                    start_month = int(iso_match.group(2))
                    start_day = int(iso_match.group(3))
                    end_year = int(iso_match.group(4))
                    end_month = int(iso_match.group(5))
                    end_day = int(iso_match.group(6))
                    
                    start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"
                    end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
            except Exception as e:
                print(f"ISO 날짜 형식 파싱 오류: {str(e)}")
        
        # 정책 타입 결정 로직
        application_deadline = policy_data.get("접수마감일자", "")
        current_date = datetime.now()
        
        policy_type = "여유있는 정책"  # 기본값
        
        # 마감일이 있고 파싱 가능한 경우
        if application_deadline:
            try:
                # 예시: "2023-12-31" 형태 파싱
                deadline_parts = application_deadline.split("-")
                if len(deadline_parts) == 3:
                    deadline = datetime(int(deadline_parts[0]), int(deadline_parts[1]), int(deadline_parts[2]))
                    days_remaining = (deadline - current_date).days
                    
                    if days_remaining < 0:
                        policy_type = "종료된 정책"
                    elif days_remaining <= 7:
                        policy_type = "마감 임박 정책"
            except:
                # 파싱 실패 시 기본값 유지
                pass
        elif end_date:
            # 접수마감일자가 없지만 파싱한 end_date가 있는 경우
            try:
                end_parts = end_date.split("-")
                if len(end_parts) == 3:
                    end_date_obj = datetime(int(end_parts[0]), int(end_parts[1]), int(end_parts[2]))
                    days_remaining = (end_date_obj - current_date).days
                    
                    if days_remaining < 0:
                        policy_type = "종료된 정책"
                    elif days_remaining <= 7:
                        policy_type = "마감 임박 정책"
            except:
                # 파싱 실패 시 기본값 유지
                pass
        
        # 중요: 원본 서비스ID를 id 필드로 정확히 유지
        service_id = policy_data.get("서비스ID", "")
        
        # 필드 매핑: 한글 필드명 -> 영어 필드명 (모든 필드 포함)
        return {
            # 필수 식별 필드
            "id": service_id,  # 원본 서비스ID 유지
            "service_id": service_id,  # 서비스ID도 별도 필드로 제공
            
            # 기본 정보
            "title": policy_data.get("서비스명", "제목 없음"),
            "content": policy_data.get("서비스목적", "내용 없음"),
            "support_type": policy_data.get("지원유형", "정보 없음"),
            
            # 지원 대상 및 내용
            "eligibility": policy_data.get("지원대상", "자격 조건 정보 없음"),
            "benefits": policy_data.get("지원내용", "혜택 정보 없음"),
            "selection_criteria": policy_data.get("선정기준", "정보 없음"),
            
            # 신청 관련 정보
            "application_period": application_period,  # 원본 신청기한 텍스트
            "application_method": policy_data.get("신청방법", "정보 없음"),
            "required_documents": policy_data.get("구비서류", "정보 없음"),
            "application_office": policy_data.get("접수기관명", "정보 없음"),
            
            # 문의 및 법령 정보
            "contact_info": policy_data.get("문의처", "정보 없음"),
            "legal_basis": policy_data.get("법령", "정보 없음"),
            "administrative_rule": policy_data.get("행정규칙", ""),  # 추가: 행정규칙
            "local_law": policy_data.get("자치법규", ""),  # 추가: 자치법규
            
            # 기관 및 상태 정보
            "responsible_agency": policy_data.get("소관기관명", ""),  # 추가: 소관기관명
            "last_updated": policy_data.get("수정일시", ""),  # 추가: 수정일시
            
            # 링크
            "online_application_url": policy_data.get("온라인신청사이트URL", ""),
            
            # 날짜 정보 - 날짜 형식으로 통일
            "startDate": start_date,  # YYYY-MM-DD 형식
            "endDate": end_date,      # YYYY-MM-DD 형식
            "type": policy_type,
            
            # 원본 데이터 필드도 유지 (필요시)
            "org_data": {
                "서비스ID": service_id,
                "서비스명": policy_data.get("서비스명", ""),
                "서비스목적": policy_data.get("서비스목적", ""),
                "신청기한": application_period,
                "지원대상": policy_data.get("지원대상", ""),
                "선정기준": policy_data.get("선정기준", ""),
                "지원내용": policy_data.get("지원내용", ""),
                "신청방법": policy_data.get("신청방법", ""),
                "구비서류": policy_data.get("구비서류", ""),
                "접수기관명": policy_data.get("접수기관명", ""),
                "문의처": policy_data.get("문의처", ""),
                "법령": policy_data.get("법령", ""),
                "온라인신청사이트URL": policy_data.get("온라인신청사이트URL", ""),
                "수정일시": policy_data.get("수정일시", ""),
                "소관기관명": policy_data.get("소관기관명", ""),
                "행정규칙": policy_data.get("행정규칙", ""),
                "자치법규": policy_data.get("자치법규", "")
            }
        }
    except Exception as e:
        print(f"정책 데이터 형식 변환 중 오류 발생: {str(e)}")
        return None

# 월별 마지막 날짜 계산 헬퍼 함수
def get_last_day_of_month(month, year=None):
    if year is None:
        year = datetime.now().year
        
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        # 윤년 계산
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29
        else:
            return 28
    return 30  # 기본값

# chat.py에서만 사용되던 간소화된 형식의 함수 추가
def format_policy_simple(policy_data):
    """
    API에서 받은 정책 정보를 간소화된 형식으로 변환합니다.
    
    Args:
        policy_data: API에서 반환된 원본 정책 데이터
        
    Returns:
        dict: 간소화된 형식의 정책 데이터
    """
    if not policy_data:
        return None
    
    try:
        # 원본 서비스ID 유지
        service_id = policy_data.get("서비스ID", "")
        
        # 한글 필드명 -> 영어 필드명 (모든 필드 포함)
        return {
            # 필수 식별 필드
            "id": service_id,  # 원본 서비스ID 유지
            "service_id": service_id,  # 서비스ID도 별도 필드로 제공
            
            # 기본 정보
            "title": policy_data.get("서비스명", "제목 없음"),
            "content": policy_data.get("서비스목적", "내용 없음"),
            "support_type": policy_data.get("지원유형", "정보 없음"),
            
            # 지원 대상 및 내용
            "eligibility": policy_data.get("지원대상", "자격 조건 정보 없음"),
            "benefits": policy_data.get("지원내용", "혜택 정보 없음"),
            "selection_criteria": policy_data.get("선정기준", "정보 없음"),
            
            # 신청 관련 정보
            "application_period": policy_data.get("신청기한", ""),
            "application_method": policy_data.get("신청방법", "정보 없음"),
            "required_documents": policy_data.get("구비서류", "정보 없음"),
            "application_office": policy_data.get("접수기관명", "정보 없음"),
            
            # 문의 및 법령 정보
            "contact_info": policy_data.get("문의처", "정보 없음"),
            "legal_basis": policy_data.get("법령", "정보 없음"),
            "administrative_rule": policy_data.get("행정규칙", ""),  # 추가: 행정규칙
            "local_law": policy_data.get("자치법규", ""),  # 추가: 자치법규
            
            # 기관 및 상태 정보
            "responsible_agency": policy_data.get("소관기관명", ""),  # 추가: 소관기관명
            "last_updated": policy_data.get("수정일시", ""),  # 추가: 수정일시
            
            # 링크
            "online_application_url": policy_data.get("온라인신청사이트URL", "")
        }
    except Exception as e:
        print(f"정책 데이터 형식 변환 중 오류 발생: {str(e)}")
        return None
