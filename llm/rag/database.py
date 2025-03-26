"""
데이터베이스 연결 및 쿼리 관련 기능을 제공하는 모듈
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from langchain.sql_database import SQLDatabase
from config import DB_URI, DB_URI_DB1, DB_URI_DB2

# 전역 변수로 DB 연결 캐싱
_db_conn1 = None
_db_conn2 = None

def get_db_conn1():
    """database1 연결 싱글톤 패턴으로 로드"""
    global _db_conn1
    if _db_conn1 is None:
        _db_conn1 = SQLDatabase.from_uri(DB_URI_DB1)
    return _db_conn1

def get_db_conn2():
    """database2 연결 싱글톤 패턴으로 로드"""
    global _db_conn2
    if _db_conn2 is None:
        _db_conn2 = SQLDatabase.from_uri(DB_URI_DB2)
    return _db_conn2

async def get_user_data(user_id: str) -> Dict[str, Any]:
    """사용자 정보 조회"""
    engine = create_engine(DB_URI_DB1)
    with engine.connect() as connection:
        result = connection.execute(text(f"SELECT area, district, birthDate, gender, incomeRange, personalCharacteristics, householdCharacteristics FROM user WHERE user_id = '{user_id}'"))
        user_data = result.fetchone()
        if user_data:
            print(f"사용자 정보: {user_data}","*"*10)
            # 사용자 정보를 딕셔너리로 변환
            return {key: value for key, value in zip(result.keys(), user_data)}
        return {}

async def execute_sql_query(sql_query: str) -> Optional[List[Tuple]]:
    """SQL 쿼리 실행"""
    try:
        print(f"SQL문 실행 : {sql_query}")
        engine = create_engine(DB_URI_DB1)
        with engine.connect() as connection:
            print("DB연결")
            result = connection.execute(text(sql_query))
            return result.fetchall()
    except Exception as e:
        print(f"SQL 실행 오류: {e}")
        return None

def is_valid_sql_format(sql_text):
    """기본적인 SQL 문법 형식 검증"""
    if not sql_text:
        return False
    
    # SELECT로 시작하는지 확인
    if not re.match(r'^\s*SELECT', sql_text, re.IGNORECASE):
        return False
    
    # FROM이 포함되어 있는지 확인
    if "FROM" not in sql_text.upper():
        return False
        
    return True

AREA_DISTRICT_MAPPING = {
    "서울특별시": [
        "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", 
        "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", 
        "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", 
        "영등포구", "용산구", "은평구", "종로구", "중구"
    ],
    "부산광역시": [
        "강서구", "금정구", "기장군", "남구", "동구", "동래구", 
        "부산진구", "북구", "사상구", "사하구", "서구", 
        "수영구", "연제구", "영도구", "중구", "해운대구"
    ],
    "대구광역시": [
        "군위군", "남구", "달서구", "달성군", "동구", "북구", 
        "서구", "수성구", "중구"
    ],
    "인천광역시": [
        "강화군", "계양구", "남동구", "동구", "미추홀구", 
        "부평구", "서구", "연수구", "옹진군", "중구"
    ],
    "광주광역시": [
        "광산구", "남구", "동구", "북구", "서구"
    ],
    "대전광역시": [
        "대덕구", "동구", "서구", "유성구", "중구"
    ],
    "울산광역시": [
        "남구", "동구", "북구", "울주군", "중구"
    ],
    "세종특별자치시": ["세종특별자치시"],
    "경기도": [
        "가평군", "고양시", "과천시", "광명시", "광주시", "구리시", 
        "군포시", "김포시", "남양주시", "동두천시", "부천시", 
        "성남시", "수원시", "시흥시", "안산시", "안성시", 
        "안양시", "양주시", "양평군", "여주시", "연천군", 
        "오산시", "용인시", "의왕시", "의정부시", "이천시", 
        "파주시", "평택시", "포천시", "하남시", "화성시"
    ],
    "충청북도": [
        "괴산군", "단양군", "보은군", "영동군", "옥천군", 
        "음성군", "제천시", "증평군", "진천군", "청주시", "충주시"
    ],
    "충청남도": [
        "계룡시", "공주시", "금산군", "논산시", "당진시", 
        "보령시", "부여군", "서산시", "서천군", "아산시", 
        "예산군", "천안시", "청양군", "태안군", "홍성군"
    ],
    "전라남도": [
        "강진군", "고흥군", "곡성군", "광양시", "구례군", 
        "나주시", "담양군", "목포시", "무안군", "보성군", 
        "순천시", "신안군", "여수시", "영광군", "영암군", 
        "완도군", "장성군", "장흥군", "진도군", "함평군", 
        "해남군", "화순군"
    ],
    "경상북도": [
        "경산시", "경주시", "고령군", "구미시", "김천시", 
        "문경시", "봉화군", "상주시", "성주군", "안동시", 
        "영덕군", "영양군", "영주시", "영천시", "예천군", 
        "울릉군", "울진군", "의성군", "청도군", "청송군", 
        "칠곡군", "포항시"
    ],
    "경상남도": [
        "거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", 
        "사천시", "산청군", "양산시", "의령군", "진주시", 
        "창녕군", "창원시", "통영시", "하동군", "함안군", 
        "함양군", "합천군"
    ],
    "제주특별자치도": [
        "서귀포시", "제주시"
    ],
    "강원특별자치도": [
        "강릉시", "고성군", "동해시", "삼척시", "속초시", 
        "양구군", "양양군", "영월군", "원주시", "인제군", 
        "정선군", "철원군", "춘천시", "태백시", "평창군", 
        "홍천군", "화천군", "횡성군"
    ],
    "전북특별자치도": [
        "고창군", "군산시", "김제시", "남원시", "무주군", 
        "부안군", "순창군", "완주군", "익산시", "임실군", 
        "장수군", "전주시", "덕진구", "완산구", 
        "정읍시", "진안군"
    ],
}


ALLOWED_VALUES = {
    "area": ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "충청북도", "충청남도", "전라남도", "경상북도", "경상남도", "제주특별자치도", "강원특별자치도", "전북특별자치도"],
    "district": ["강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구", "군위군", "남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구", "강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구", "광산구", "남구", "동구", "북구", "서구", "대덕구", "동구", "서구", "유성구", "중구", "남구", "동구", "북구", "울주군", "중구", "세종특별자치시", "가평군", "고양시", "과천시", "광명시", "광주시", "구리시", "군포시", "김포시", "남양주시", "동두천시", "부천시", "성남시", "수원시", "시흥시", "안산시", "안성시", "안양시", "양주시", "양평군", "여주시", "연천군", "오산시", "용인시", "의왕시", "의정부시", "이천시", "파주시", "평택시", "포천시", "하남시", "화성시", "괴산군", "단양군", "보은군", "영동군", "옥천군", "음성군", "제천시", "증평군", "진천군", "청주시", "충주시", "계룡시", "공주시", "금산군", "논산시", "당진시", "보령시", "부여군", "서산시", "서천군", "아산시", "예산군", "천안시", "청양군", "태안군", "홍성군", "강진군", "고흥군", "곡성군", "광양시", "구례군", "나주시", "담양군", "목포시", "무안군", "보성군", "순천시", "신안군", "여수시", "영광군", "영암군", "완도군", "장성군", "장흥군", "진도군", "함평군", "해남군", "화순군", "경산시", "경주시", "고령군", "구미시", "김천시", "문경시", "봉화군", "상주시", "성주군", "안동시", "영덕군", "영양군", "영주시", "영천시", "예천군", "울릉군", "울진군", "의성군", "청도군", "청송군", "칠곡군", "포항시", "거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", "사천시", "산청군", "양산시", "의령군", "진주시", "창녕군", "창원시", "통영시", "하동군", "함안군", "함양군", "합천군", "서귀포시", "제주시", "강릉시", "고성군", "동해시", "삼척시", "속초시", "양구군", "양양군", "영월군", "원주시", "인제군", "정선군", "철원군", "춘천시", "태백시", "평창군", "홍천군", "화천군", "횡성군", "고창군", "군산시", "김제시", "남원시", "무주군", "부안군", "순창군", "완주군", "익산시", "임실군", "장수군", "전주시", "전주시 덕진구", "전주시 완산구", "정읍시", "진안군"],
    "gender": ["남자", "여자"],
    "income_category": ["0 ~ 50%", "51 ~ 75%", "76 ~ 100%", "101 ~ 200%"],
    "personal_category": ["예비부부/난임", "임신부", "출산/입양", "장애인", "국가보훈대상자", "농업인", "어업인", "축산인", "임업인", "초등학생", "중학생", "고등학생", "대학생/대학원생", "질병/질환자", "근로자/직장인", "구직자/실업자", "해당사항 없음"],
    "household_category": ["다문화가정", "북한이탈주민가정", "한부모가정/조손가정", "1인 가구", "다자녀가구", "무주택세대", "신규전입가구", "확대가족", "해당사항 없음"],
    "support_type": ["현금", "현물", "서비스", "이용권"],
    "application_method": ["온라인 신청", "타사이트 신청", "방문 신청", "기타"],
    "benefit_category": ["생활안정", "주거-자립", "보육-교육", "고용-창업", "보건-의료", "행정-안전", "임신-출산", "보호-돌봄", "문화-환경", "농림축산어업"]
}


async def is_valid_sql(sql_query: str) -> bool:
    """SQL 쿼리가 유효한지 확인"""
    try:
        engine = create_engine(DB_URI_DB1)
        with engine.connect() as connection:
            # SQL 구문 검증만 수행
            connection.execute(text(sql_query))
            return True
    except Exception as e:
        print(f"SQL 유효성 검사 실패: {e}")
        return False

async def replace_select_with_star_indexing(sql_query: str) -> str:
    try:
        select_index = sql_query.index("SELECT")
        from_index = sql_query.index("FROM")
        start_index = select_index + len("SELECT")
        end_index = from_index
        modified_query = sql_query[:start_index] + " *" + sql_query[end_index:]
        return modified_query
    except ValueError:
        return "SELECT 또는 FROM 키워드를 찾을 수 없습니다."

async def fetch_documents_by_ids(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """문서 ID 리스트로부터 실제 문서 내용 조회"""
    engine = create_engine(DB_URI_DB1)
    result_docs = []
    
    with engine.connect() as connection:
        for doc_id in doc_ids:
            result = connection.execute(text(f"SELECT * FROM benefits WHERE service_id = '{doc_id}'"))
            doc = result.fetchone()
            if doc:
                # 문서 정보를 딕셔너리로 변환
                doc_dict = {key: value for key, value in zip(result.keys(), doc)}
                result_docs.append(doc_dict)
    
    return result_docs

def ensure_service_id_in_sql(sql_query: str) -> str:
    """SQL 쿼리에 service_id가 포함되어 있는지 확인하고 없으면 추가"""
    if "service_id" not in sql_query.lower() and "select" in sql_query.lower():
        # service_id 추가
        select_pattern = re.compile(r'(SELECT\s+)(.*?)(\s+FROM)', re.IGNORECASE | re.DOTALL)
        if select_pattern.search(sql_query):
            return select_pattern.sub(r'\1service_id, \2\3', sql_query)
    return sql_query

# def extract_sql_from_text(text: str) -> str:
#     """텍스트에서 SQL 쿼리 추출"""
#     sql_pattern = re.compile(r'```sql\s*(.*?)\s*```|```\s*(.*?)\s*```|SELECT\s+.*?;', re.DOTALL)
#     match = sql_pattern.search(text)
#     if match:
#         clean_sql = match.group(1) or match.group(2) or match.group(0)
#         return clean_sql.strip().rstrip(';') + ';'
#     return text

def get_db_connection():
    """데이터베이스 연결 생성"""
    try:
        # 실제 데이터베이스 이름으로 변경
        connection = create_engine('mysql+mysqlconnector://username:password@localhost/multimodal_final_project')
        return connection
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        # 오류 발생 시 대체 로직 구현
        return None