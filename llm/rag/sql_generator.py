"""
SQL 쿼리 생성 및 처리 관련 기능을 제공하는 모듈
"""
import os
import re
import time
import traceback
import asyncio
import datetime
import requests
from typing import List, Dict, Any, Tuple, Set, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from user_data import combine_user_data
from config import MAX_SQL_ATTEMPTS, MISTRAL_VLLM
from database import (
    get_user_data, 
    execute_sql_query, 
    is_valid_sql, 
    is_valid_sql_format,
    ensure_service_id_in_sql,
    replace_select_with_star_indexing
)
from embedding import get_question_vectorstore

# 허용된 값 목록 정의
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


def extract_sql_from_text(text: str) -> str:
    # SQL 태그 검색
    sql_match = re.search(r"<SQL>(.*?)</SQL>", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    # 태그가 없으면 일반 SQL 키워드로 시작하는 부분 찾기
    sql_keywords = ["SELECT", "WITH", "CREATE", "INSERT", "UPDATE", "DELETE"]
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped_line = line.strip().upper()
        if any(stripped_line.startswith(keyword) for keyword in sql_keywords):
            return "\n".join(lines[i:]).strip()
    
    return text.strip()

def validate_sql_categories(sql_query):
    """SQL 쿼리에 사용된 카테고리 값이 허용된 값인지 검증"""
    # 각 허용된 필드에 대해 검사
    for field, allowed_values in ALLOWED_VALUES.items():
        # 정규 표현식으로 필드 비교 연산 찾기 (예: field = 'value' 또는 field IN ('value1', 'value2'))
        matches = re.finditer(r'{}[\s]*=[\s]*[\'\"](.*?)[\'\"]'.format(field), sql_query, re.IGNORECASE)
        for match in matches:
            value = match.group(1)
            if value != "" and value not in allowed_values:
                return False, f"허용되지 않은 {field} 값: {value}"
        
        # IN 구문 검사
        in_matches = re.finditer(r'{}[\s]+IN[\s]*\((.*?)\)'.format(field), sql_query, re.IGNORECASE)
        for match in in_matches:
            values_str = match.group(1)
            values = [v.strip().strip('\'"') for v in values_str.split(',')]
            for value in values:
                if value != "" and value not in allowed_values:
                    return False, f"허용되지 않은 {field} 값: {value}"
    
    return True, ""

def clean_sql_query(sql_query):
    """SQL 쿼리에서 문제가 될 수 있는 요소 제거"""
    # 주석 제거
    sql_query = re.sub(r'--.*?(\n|$)', '', sql_query)
    
    # LIMIT 구문 제거
    sql_query = re.sub(r'\bLIMIT\s+\d+\s*($|;)', '', sql_query, flags=re.IGNORECASE)
    
    # 테이블 존재 검증 (benefits 테이블만 허용)
    if re.search(r'FROM\s+(?!benefits\b)[a-zA-Z_][a-zA-Z0-9_]*', sql_query, re.IGNORECASE):
        # benefits 외 다른 테이블이 FROM 절에 있으면 수정
        sql_query = re.sub(r'FROM\s+(?!benefits\b)[a-zA-Z_][a-zA-Z0-9_]*', 'FROM benefits', sql_query, flags=re.IGNORECASE)
    
    # JOIN 구문 제거 (필요한 경우 더 정교한 방식으로 수정)
    sql_query = re.sub(r'LEFT\s+JOIN.*?ON.*?(?=WHERE|GROUP|ORDER|LIMIT|$)', '', sql_query, flags=re.IGNORECASE | re.DOTALL)
    sql_query = re.sub(r'JOIN.*?ON.*?(?=WHERE|GROUP|ORDER|LIMIT|$)', '', sql_query, flags=re.IGNORECASE | re.DOTALL)
    
    # 복잡한 CASE/IF 구문 단순화 (보수적인 접근)
    if 'CASE' in sql_query.upper() or 'IF(' in sql_query.upper():
        # CASE/IF 구문이 SELECT 필드에만 있는지 확인
        if re.search(r'(CASE|IF\().*?(FROM)', sql_query, re.IGNORECASE | re.DOTALL):
            # SELECT와 FROM 사이의 텍스트 추출
            select_part = re.search(r'SELECT(.*?)FROM', sql_query, re.IGNORECASE | re.DOTALL)
            if select_part:
                # service_id가 포함되었는지 확인
                has_service_id = 'service_id' in select_part.group(1)
                # 복잡한 CASE/IF 구문을 단순 컬럼 선택으로 변경
                simplified_select = 'SELECT ' + ('service_id, ' if not has_service_id else '') + 'area, district, min_age, max_age, gender, benefit_category, support_type'
                sql_query = re.sub(r'SELECT.*?FROM', simplified_select + ' FROM', sql_query, flags=re.IGNORECASE | re.DOTALL)
    
    return sql_query

async def generate_sql_query(question: str, propmt_str: str) -> Optional[str]:
    step_timings = {}
    overall_start = time.time()
    print("SQL 쿼리 생성 및 검증")
     
    # 상세한 스키마 정의 - 허용 값 명확히 표시
    detailed_schema = f"""
Database: multimodal_final_project
Database Schema:
Table: benefits
Columns:
- service_id: 서비스 고유 ID (문자열)
- area: 혜택이 제공되는 광역 행정 구역 (유효값: "전국", "" 또는 {ALLOWED_VALUES["area"]} 중 하나)
- district: 혜택이 제공되는 기초 행정 구역 (유효값: "" 또는 정해진 목록 중 하나, 중복 가능)
- min_age: 혜택을 받을 수 있는 최소 나이 (숫자로만 출력)
- max_age: 혜택을 받을 수 있는 최대 나이 (숫자로만 출력)
- gender: 혜택을 받을 수 있는 성별 (유효값: {ALLOWED_VALUES["gender"]})
- income_category: 혜택을 받을 수 있는 소득 백분률 분류 (유효값: "" 또는 {ALLOWED_VALUES["income_category"]} 중 하나)
- personal_category: 혜택 지원 대상인 개인의 특성 분류 (유효값: "" 또는 {ALLOWED_VALUES["personal_category"]} 중에서 선택, 중복 가능)
- household_category: 혜택 대상의 가구 유형 카테고리 (유효값: "" 또는 {ALLOWED_VALUES["household_category"]} 중에서 선택, 중복 가능)
- support_type: 혜택 지원 유형 분류 (유효값: {ALLOWED_VALUES["support_type"]} 중 하나)
- application_method: 혜택 신청 방법 분류 (유효값: {ALLOWED_VALUES["application_method"]} 중 하나)
- benefit_category: 혜택이 속하는 카테고리 분류 (유효값: {ALLOWED_VALUES["benefit_category"]} 중 하나)
- start_date: 혜택 신청 시작 날짜 (YY-MM-DD 형식)
- end_date: 혜택 신청 종료 날짜 (YY-MM-DD 형식)
- date_summary: start_date, end_date를 YY-MM-DD 형식으로 요약
- source: 혜택 정보 출처
"""

    prompt_template = PromptTemplate(
        template=propmt_str,
        input_variables=["schema", "question"]
    )
    
    prompt_text = prompt_template.format(schema=detailed_schema, question=question)
    
    # generate_sql_query 함수에서 데이터 부분 수정
    data = {
        "prompt": prompt_text,
        "max_tokens": 4048,
        "stop": ["</SQL>"]  # SQL 완성 후 중단
    }

    max_attempts = 5
    attempt = 0
    valid = False
    modified_query = None

    while attempt < max_attempts and not valid:
        print(f"시도 {attempt+1} 시작")
        try:
            response = requests.post(
                url=MISTRAL_VLLM, 
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=45
            )
            response.raise_for_status()
            
            result = response.json()
            print("result : ", result)
            
            # text 배열의 첫 번째 요소에서 SQL 쿼리 추출
            if 'text' in result and result['text'] and isinstance(result['text'], list):
                full_response_text = result['text'][0]
                print("full_response_text : ", full_response_text)
                
                # SQL 쿼리 부분만 추출 (커스텀 태그로 변경)
                sql_matches = re.search(r"<SQL>(.*?)</SQL>", full_response_text, re.DOTALL)
                                
                if sql_matches:
                    sql_query = sql_matches.group(1).strip()
                    print("추출된 SQL 쿼리:", sql_query)
                else:
                    # 백업 방식: 마크다운 형식 찾기
                    sql_matches = re.search(r"```sql\s*(.*?)(```|$)", full_response_text, re.DOTALL)
                    if sql_matches:
                        sql_query = sql_matches.group(1).strip()
                        print("마크다운에서 추출된 SQL 쿼리:", sql_query)
                    else:
                        print("SQL 쿼리를 찾을 수 없습니다")
                        attempt += 1
                        continue  # 재시도
            
            print("추출 전 LLM 출력:")
            print(sql_query)
            
            # SQL이 None이 아닌지 확인 후 처리
            if sql_query is None:
                print("SQL 쿼리가 None입니다")
                attempt += 1
                continue  # 재시도
            
            # SQL 쿼리 추출 및 검증
            step_start = time.time()
            clean_sql = extract_sql_from_text(sql_query)
            clean_sql = clean_sql_query(clean_sql) 
            step_end = time.time()
            step_timings["SQL 추출"] = step_end - step_start

            print("추출된 SQL:")
            print(clean_sql)
            
            is_valid_categories, error_msg = validate_sql_categories(clean_sql)
            if not is_valid_sql_format(clean_sql):
                print("유효하지 않은 SQL 형식")
                attempt += 1
                continue  # 재시도
                
            # service_id가 SQL에 포함되어 있는지 확인 후 추가 
            if not re.search(r"service_id", clean_sql, re.IGNORECASE):
                clean_sql = ensure_service_id_in_sql(clean_sql)
                print("service_id 추가 후 SQL:", clean_sql)
            
            step_start = time.time()
            modified_query = await replace_select_with_star_indexing(clean_sql)
            step_end = time.time()
            step_timings["SELECT 치환"] = step_end - step_start

            # SQL 유효성 최종 검증
            step_start = time.time()
            valid = await is_valid_sql(modified_query)
            step_end = time.time()
            step_timings["SQL 검증"] = step_end - step_start

            if valid:
                print(f"유효한 SQL 생성: {modified_query}")
            else:
                print(f"유효하지 않은 SQL: {modified_query}")
                attempt += 1
                continue  # 재시도
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            attempt += 1
            continue  # 재시도
        except Exception as e:
            print(f"SQL 생성 중 에러 발생: {e}")
            attempt += 1
            continue  # 재시도

    overall_end = time.time()
    step_timings["전체 SQL 생성"] = overall_end - overall_start

    print("\n[SQL 생성 단계별 소요 시간]")
    for step, duration in step_timings.items():
        print(f"{step}: {duration:.2f} 초")

    if not valid:
        print("최대 시도 횟수 도달: 유효한 SQL 생성 실패")
        return None

    print("최종 SQL 쿼리:", modified_query)
    return modified_query

async def get_prompt_with_fewshot_example(question: str) -> Dict:
    """
    Few-shot 예제를 생성하고, 생성에 소요된 시간을 측정하여 출력합니다.
    """
    t_start = time.time()
    
    # 기본 예제 정의 (벡터스토어 로드 실패 시 사용)
    default_examples = [
        {
            "query": "서울시에 사는 30대 남성을 위한 주거 지원 혜택이 있나요?",
            "generated_sql": "SELECT service_id FROM benefits WHERE area = '서울특별시' AND min_age <= 30 AND max_age >= 39 AND gender = '남자' AND benefit_category = '주거-자립'"
        },
        {
            "query": "경기도 거주 대학생 장학금 지원 프로그램 알려주세요",
            "generated_sql": "SELECT service_id FROM benefits WHERE area = '경기도' AND personal_category = '대학생/대학원생' AND benefit_category = '보육-교육'"
        }
    ]
    
    try:
        # 벡터스토어 로드
        vectorstore = get_question_vectorstore()
        
        if vectorstore is not None:
            # 벡터스토어가 정상적으로 로드된 경우
            similar_examples = vectorstore.similarity_search(question, k=2)
            example1 = {
                "query": similar_examples[0].page_content,
                "generated_sql": similar_examples[0].metadata.get("sql_query", "")
            }
            example2 = {
                "query": similar_examples[1].page_content,
                "generated_sql": similar_examples[1].metadata.get("sql_query", "")
            }
        else:
            # 벡터스토어 로드 실패 시 기본 예제 사용
            print("벡터스토어 로드 실패: 기본 예제 사용")
            example1 = default_examples[0]
            example2 = default_examples[1]
    except Exception as e:
        print(f"Few-shot 예제 생성 중 오류 발생: {e}")
        # 오류 발생 시 기본 예제 사용
        example1 = default_examples[0]
        example2 = default_examples[1]
    
    # 프롬프트 구성 
    prompt_str = f"""### 요구 사항:
아래 질문을 SQL 쿼리로 변환하세요.

### Database Schema:
{{schema}}

### 쿼리 생성 규칙:
1. 각 컬럼에 대한 제약사항을 반드시 지켜야 합니다. 스키마에 없는 컬럼, 테이블, 값을 사용하지 말고, 잘못된 데이터 타입이나 비현실적인 조건을 생성하지 마세요.
2. 결과 필드는 질문에서 요구하는 정보만 포함하세요. 질문에 명시된 내용과 직접 관련된 스키마의 컬럼만 선택하고, 불필요한 컬럼은 추가하지 마세요.
3. LIKE 연산자를 사용할 때는 '%' 와일드카드를 적절히 활용하세요. 예: 부분 문자열 검색 시 `WHERE column LIKE '%keyword%'`, 시작 문자열 검색 시 `WHERE column LIKE 'keyword%'`.
4. 날짜는 'YYYY-MM-DD' 형식으로 입력하세요 (예: '2023-10-01'). 스키마에 저장된 날짜 형식과 일치하도록 주의하세요.
5. 카테고리 필드는 스키마에 정의된 값 목록(예: '생활안정', '교육', '건강') 내에서만 검색하세요. 목록은 스키마를 참조하세요.
6. SQL 쿼리에 주석(예: -- 또는 /* */)을 포함하지 마세요. 주석은 쿼리 실행 시 오류를 유발할 수 있습니다.
7. 스키마에 정의된 테이블과 컬럼만 사용하세요. 여러 테이블의 정보를 결합해야 할 때만 JOIN을 사용하고, 스키마에 없는 테이블이나 컬럼은 절대 참조하지 마세요.
8. 복잡한 CASE/IF 문은 사용하지 마세요. 단순 비교 연산자(예: =, <, >)로 해결 가능한 경우에는 해당 연산자를 우선 사용하세요. 예외적으로 여러 조건에 따라 값을 변환해야 할 때만 사용하세요.
9. 조건 값이나 컬럼명에 임의의 한국어 텍스트를 생성하지 마세요. 반드시 스키마에 정의된 값만 사용하세요 (예: '서울특별시'는 사용 가능, '임의 지역'은 불가).

### 예시 SQL:
질문 : {example1["query"]}
<SQL>{example1["generated_sql"]}</SQL>

질문 : {example2["query"]}
<SQL>{example2["generated_sql"]}</SQL>

### 질문:
{question}

### SQL Query:
<SQL>
"""
    t_end = time.time()
    elapsed = t_end - t_start
    print(f"[타이밍] Few-shot 예제 생성: {elapsed:.2f} 초")
    return {"prompt": prompt_str, "elapsed": elapsed}

async def get_sql_results(question: str, user_id: str, user_data: Optional[Dict[str, Any]] = None) -> List[str]:
    """정보2 수집: Text-to-SQL을 통한 문서 ID 검색"""
    print("정보2 수집 시작...")
    try:
        # 이미 user_data가 전달되었으면 재호출하지 않음
        if user_data is None:
            user_data = await get_user_data(user_id)
        print(f"사용자 정보: {user_data}")
        
        # Few-shot 예제 생성 및 시간 측정 추가
        fewshot_start = time.time()
        examples = await get_prompt_with_fewshot_example(question)
        fewshot_end = time.time()
        print(f"[타이밍] Few-shot 예제 생성 (get_sql_results 내부): {fewshot_end - fewshot_start:.2f} 초")
        
        sql_query = await generate_sql_query(question, examples["prompt"])
        if not sql_query:
            print("SQL 쿼리 생성 실패")
            return []
            
        sql_query = await combine_user_data(sql_query, user_data)
        results = await execute_sql_query(sql_query)
        
        service_ids = []
        if results:
            service_ids = [result.service_id for result in results if hasattr(result, 'service_id')]
        
        print(f"정보2 수집 완료: {len(service_ids)}개 service_id 수집")
        return service_ids
    except Exception as e:
        print(f"get_sql_results 에러 발생: {e}")
        return []