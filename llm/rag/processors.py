"""
데이터 처리 관련 기능을 제공하는 모듈
"""
import os
import asyncio
import datetime
import re
import requests
import time
from typing import List, Dict, Any, Tuple, Set, Optional
from langchain.schema import Document
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from config import TOP_K_INITIAL, TOP_K_RERANK, MAX_SQL_ATTEMPTS, FINAL_DOCS_COUNT, EMBEDDING_MODEL, MISTRAL_VLLM, JSON_FILE_PATH, VECTORSTORE_A_PATH
from dotenv import load_dotenv 
from models import (get_llm, load_tunned_model)
from vectorstore import (
    load_vectorstore_a, 
    # load_vectorstore_c, 
    retrieve_similar_docs,
    #fetch_documents_by_ids,
    create_ensemble_retriever
)
from database import (
    get_user_data, 
    execute_sql_query, 
    is_valid_sql, 
    is_valid_sql_format,
    ensure_service_id_in_sql,
    replace_select_with_star_indexing
)
from prompts import (
    get_text_to_sql_prompt, 
    get_final_answer_prompt
)
from step_back_rerank import(
    StepBackRAG,
    generate_step_back_query
)


_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model

RAG = StepBackRAG(vectorstore_path=VECTORSTORE_A_PATH, json_data_path=JSON_FILE_PATH)

def rerank_documents_and_extract_service_ids(documents: List[Document], question: str, top_k: int = 15) -> List[str]:
    load_dotenv()
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    """Re-ranking을 통해 문서 목록을 정렬한 후, 각 문서의 서비스ID 리스트를 반환"""
    if not COHERE_API_KEY:
        print("COHERE_API_KEY가 설정되지 않았습니다.")
        return [doc.metadata.get("서비스ID", "미확인") for doc in documents[:top_k]]
    
    try:
        compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_k, cohere_api_key=COHERE_API_KEY)

        # FAISS 벡터 스토어 생성 (임시)
        embedding_model = get_embedding_model()
        vectorstore = FAISS.from_documents(documents, embedding_model)

        # retriever 객체는 vectorstore의 as_retriever() 메서드를 통해 생성합니다.
        base_retriever = vectorstore.as_retriever()

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        # 질문에 따른 압축 및 점수 계산 결과를 얻음
        compressed_docs = compression_retriever.invoke(question)

        print(f"compressed_docs 길이: {len(compressed_docs)}") # 디버깅용 출력 추가
        
        # 압축된 문서에서 서비스ID 추출
        service_ids = [doc.metadata.get("서비스ID", "미확인") for doc in compressed_docs[:top_k]]
        print(f"service_ids 길이: {len(service_ids)}") # 디버깅용 출력 추가

        if not service_ids:
            return []  # 빈 리스트 반환
            
        return service_ids
    except Exception as e:
        print(f"rerank_documents_and_extract_service_ids 에러 발생: {e}")
        # 에러 발생 시 원본 문서에서 서비스ID 추출
        return [doc.metadata.get("서비스ID", "미확인") for doc in documents[:top_k]]

def extract_sql_from_text(text: str) -> str:
    """텍스트에서 SQL 쿼리 부분만 추출"""
    if not text:
        return ""
        
    # SQL 코드 블록 추출 (```sql ... ``` 형식)
    sql_match = re.search(r"```sql\s+(.*?)\s*```", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    # SQL 키워드로 시작하는 라인 추출
    sql_keywords = ["SELECT", "WITH", "CREATE", "INSERT", "UPDATE", "DELETE"]
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped_line = line.strip().upper()
        if any(stripped_line.startswith(keyword) for keyword in sql_keywords):
            return "\n".join(lines[i:]).strip()
    
    return text.strip()

def convert_dicts_to_documents(docs_dict: List[Dict[str, Any]]) -> List[Document]:
    """딕셔너리 리스트에서 Document 객체 리스트로 변환"""
    documents = []
    for doc in docs_dict:
        support_detail = doc.get('지원내용', '') or ''
        service_summary = doc.get('서비스목적요약', '') or ''
        service_category = doc.get("서비스분야") or ''
        required_summary = doc.get("선정기준") or ''
        if not (support_detail.strip() or service_summary.strip() or service_category.strip() or required_summary.strip()):
            continue
        
        page_content = f"{support_detail}\n{service_summary}\n{service_category}\n{required_summary}".strip()
        
        document = Document(
            page_content=page_content,
            metadata={
                '서비스ID': doc.get('서비스ID', ''),
                '서비스명': doc.get('서비스명', ''),
            }
        )
        documents.append(document)
    return documents

async def get_similarity_results(question: str, stepback_question: str) -> List[str]:
    """정보1 수집: A 벡터스토어에서 문서 검색 및 Re-ranking"""
    print("정보1 수집 시작...")
    
    try:
        # Stepback을 통해 문서 검색
        docs = RAG.retrieve_with_step_back(question, stepback_question)

        # 문서 ID만 추출하여 반환
        doc_ids = []
        for doc in docs:
            doc_id = doc.metadata.get("서비스ID")
            if doc_id:  # None이나 빈 문자열이 아닌 경우만 추가
                doc_ids.append(doc_id)
                
        print(f"정보1 수집 완료: {len(doc_ids)}개 문서 ID 수집")
        return doc_ids
    except Exception as e:
        print(f"get_similarity_results 에러 발생: {e}")
        return []

async def combine_user_data(sql_query: str, user_data: Dict[str, Any]) -> str:
    # 사용자 데이터가 비어있거나 None이면 원래 쿼리 그대로 반환
    if not user_data:
        return sql_query
    
    try:    
        today = datetime.date.today()
        
        # birthDate가 이미 datetime 객체이면 바로 사용, 아니면 파싱
        birthdate_val = user_data.get("birthDate")
        if not birthdate_val:  # birthDate가 없는 경우
            return sql_query
            
        if isinstance(birthdate_val, str):
            birthdate = datetime.datetime.strptime(birthdate_val, "%Y-%m-%d").date()
        elif isinstance(birthdate_val, datetime.date):
            birthdate = birthdate_val  # 이미 date 객체이므로 그대로 사용
        else:
            # 다른 타입인 경우 (datetime.datetime 등)
            try:
                birthdate = birthdate_val.date()
            except AttributeError:
                # date() 메서드가 없으면 변환 불가, 원래 쿼리 반환
                return sql_query
            
        age = today.year - birthdate.year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            age -= 1

        additional_conditions = {}

        # 예시: district 조건은 쿼리에 없으면 추가
        if 'district' in user_data and not re.search(r"\bdistrict\b", sql_query, flags=re.IGNORECASE):
            additional_conditions["district"] = f"'{user_data['district']}'"

        # 만약 쿼리에 min_age 조건이 없으면, 사용자 나이보다 크거나 같은 조건 추가
        if not re.search(r"\bmin_age\b", sql_query, flags=re.IGNORECASE):
            additional_conditions["min_age"] = f"{age}"  # 숫자 조건

        # 만약 쿼리에 max_age 조건이 없으면, 사용자 나이보다 작거나 같은 조건 추가
        if not re.search(r"\bmax_age\b", sql_query, flags=re.IGNORECASE):
            additional_conditions["max_age"] = f"{age}"  # 숫자 조건

        # 추가 조건 문자열 구성
        condition = ""
        for column, value in additional_conditions.items():
            if column == "min_age":
                # min_age 조건: min_age <= {age}
                condition += f" AND {column} <= {value}"
            elif column == "max_age":
                # max_age 조건: max_age >= {age}
                condition += f" AND {column} >= {value}"
            else:
                # 나머지 문자열 조건
                condition += f" AND {column} = {value}"

        # WHERE 절이 있는지 확인
        if "WHERE" not in sql_query.upper() and condition:
            # WHERE 절이 없을 경우 추가
            where_position = sql_query.upper().find("FROM") + 4
            from_clause_end = 0
            for keyword in ["ORDER BY", "GROUP BY", "LIMIT"]:
                pos = sql_query.upper().find(keyword)
                if pos > 0:
                    from_clause_end = pos
                    break
            
            if from_clause_end > 0:
                combined_query = sql_query[:from_clause_end] + " WHERE 1=1" + condition + " " + sql_query[from_clause_end:]
            else:
                combined_query = sql_query.rstrip(";") + " WHERE 1=1" + condition + ";"
        else:
            # LIMIT 절이 포함되어 있다면, LIMIT 앞에 조건을 삽입
            if "LIMIT" in sql_query.upper():
                parts = re.split(r"(LIMIT\s+\d+)", sql_query, flags=re.IGNORECASE)
                # parts[0]: WHERE 절 등 앞부분, parts[1]: LIMIT 절, parts[2]: 나머지 (존재할 경우)
                combined_query = parts[0] + condition + " " + parts[1]
                if len(parts) > 2:
                    combined_query += " " + parts[2]
            else:
                combined_query = sql_query.rstrip(";") + condition + ";"
        
        print(f"고객정보랑 합쳐서 sql 출력: {combined_query}")
        return combined_query
    except Exception as e:
        print(f"combine_user_data 에러 발생: {e}")
        return sql_query  # 에러 발생 시 원래 쿼리 반환

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

async def generate_sql_query(question: str) -> Optional[str]:
    step_timings = {}
    overall_start = time.time()
    print("SQL 쿼리 생성 및 검증")
    
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
    # 개선된 프롬프트 템플릿
    prompt_str = """### Convert this question into SQL query:

### Database Schema:
{schema}

### 쿼리 생성 규칙:
1. 각 컬럼에 대한 제약사항을 반드시 지켜야 합니다.
2. 결과 필드는 질문에서 요구하는 내용에 따라 선택적으로 포함하되, 항상 스키마에 명시된 컬럼 이름만 사용하세요.
3. LIKE 연산자를 사용할 때는 필요에 따라 '%' 와일드카드를 적절히 사용하세요.
4. 날짜 형식은 반드시 YY-MM-DD 형식을 준수해야 합니다.
5. 카테고리 필드는 반드시 정해진 값 목록 내에서만 검색해야 합니다.
6. SQL 쿼리에 주석(comment)을 포함하지 마세요.
7. 필요한 경우에만 JOIN을 사용하고, 스키마에 없는 테이블은 절대 참조하지 마세요.
8. 복잡한 CASE/IF 문은 사용하지 마세요. 꼭 필요한 경우가 아니면 단순 비교 연산자를 사용하세요.
9. 스키마에 명시되지 않은 컬럼명이나 테이블명을 만들지 마세요.
10. 반드시 스키마에 정의된 값 목록 내에서만 비교 연산을 수행하세요.
11. 한국어 텍스트를 임의로 생성하지 마세요. 스키마에 정의된 값만 사용하세요.

### 예시 - 올바른 SQL 쿼리:
```sql
SELECT service_id, area, district, benefit_category, support_type 
FROM benefits 
WHERE area = '서울특별시' 
AND min_age <= 30 
AND max_age >= 30 
AND gender = '남자' 
AND benefit_category = '생활안정'

### 예시 - 잘못된 SQL 쿼리: 
-- 이런 주석은 포함하지 마세요
SELECT b.*, '맞춤형혜택' as custom_name  -- 임의의 한국어 텍스트 생성 금지
FROM benefits b
LEFT JOIN support_types s  -- 스키마에 없는 테이블 사용 금지
ON b.support_type = s.type
WHERE b.benefit_category = '표준자금'  -- 허용되지 않은 카테고리값 사용 금지

### Question:
{question}

### SQL Query:
```sql
"""

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["schema", "question"]
    )
    
    prompt_text = prompt_template.format(schema=detailed_schema, question=question)

    data = {
        "prompt": prompt_text,
        "max_tokens": 4048,
        "stop": ["```"]
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
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # text 배열의 첫 번째 요소에서 SQL 쿼리 추출
            if 'text' in result and result['text'] and isinstance(result['text'], list):
                full_response_text = result['text'][0]
                
                # SQL 쿼리 부분만 추출 (```sql과 ``` 사이의 내용)
                sql_matches = re.search(r"```sql\s*(.*?)(```|$)", full_response_text, re.DOTALL)
                
                if sql_matches:
                    sql_query = sql_matches.group(1).strip()
                    print("추출된 SQL 쿼리:", sql_query)
                else:
                    print("SQL 쿼리를 찾을 수 없습니다")
                    attempt += 1
                    continue  # 재시도
            else:
                print("API 응답에 text 필드가 없거나 형식이 맞지 않습니다")
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
       
async def get_sql_results(question: str, user_id: str) -> List[str]:
    """정보2 수집: Text-to-SQL을 통한 문서 ID 검색"""
    print("정보2 수집 시작...")
    
    try:
        # 사용자 정보 조회
        user_data = await get_user_data(user_id)
        print(f"사용자 정보: {user_data}")
        
        sql_query = await generate_sql_query(question)
        if not sql_query:
            print("SQL 쿼리 생성 실패")
            return []
            
        sql_query = await combine_user_data(sql_query, user_data)
        
        # 생성된 SQL로 benefits 테이블 조회
        results = await execute_sql_query(sql_query)
        
        service_ids = []
        if results:
            # service_id 추출
            service_ids = [result.service_id for result in results if hasattr(result, 'service_id')]
        
        print(f"정보2 수집 완료: {len(service_ids)}개 service_id 수집")
        return service_ids
    except Exception as e:
        print(f"get_sql_results 에러 발생: {e}")
        return []

async def supervisor(question: str, user_id: str) -> Tuple[List[Document], Dict[str, Any], Set, List, List]:
   try:
       load_dotenv()
       # 환경 변수에서 Hugging Face 토큰 가져오기
       hf_token = os.getenv("HUGGINGFACE_TOKEN")

       # 토큰 유효성 검사
       if not hf_token:
           print("HUGGINGFACE_TOKEN이 설정되지 않았습니다. 기본 모델을 사용합니다.")
           
       # Hugging Face Hub에 로그인 (토큰이 있는 경우에만)
       if hf_token:
           login(token=hf_token)
           
       llm = get_llm()
       print(f"step back 전 질문 : {question}")
       
       # stepback 질문 생성 중 오류 발생에 대비
       try:
           stepback_question = generate_step_back_query(question, llm)
           print(f"step back 후 질문 : {stepback_question}")
       except Exception as e:
           print(f"Step back 질문 생성 중 오류 발생: {e}")
           stepback_question = question  # 오류 발생 시 원래 질문 사용
       
       # 정보1과 정보2를 병렬로 수집
       info1_task = get_similarity_results(question, stepback_question)
       info2_task = get_sql_results(stepback_question, user_id)

       info1, info2 = await asyncio.gather(info1_task, info2_task)

       # 사용자 정보 조회
       user_data = await get_user_data(user_id)

       print(f"유사도 검색: {len(info1)}개 문서 ID, SQL 검색: {len(info2)}개 service_id")

       # 그룹 분리: 공통, info1-only, info2-only
       common_ids = set(info1).intersection(set(info2))
       info1_only_ids = [doc_id for doc_id in info1 if doc_id not in common_ids]
       info2_only_ids = [doc_id for doc_id in info2 if doc_id not in common_ids]
       print(f"공통 문서 ID: {common_ids}")
       print(f"백터검색 전용 문서 ID: {info1_only_ids}")
       print(f"sql쿼리 전용 문서 ID: {info2_only_ids}")

       # 벡터스토어 로드 중 오류 처리
       try:
           vectorstore = load_vectorstore_a()  # FAISS 벡터스토어 로드
           
           # 대상 문서 ID 목록
           doc_ids_to_fetch = list(common_ids) + info1_only_ids + info2_only_ids
           
           # 빈 문서 ID 제거
           doc_ids_to_fetch = [doc_id for doc_id in doc_ids_to_fetch if doc_id]
           
           if not doc_ids_to_fetch:
               print("검색된 문서 ID가 없습니다.")
               return [], user_data, common_ids, info1_only_ids, info2_only_ids
               
           all_docs = []
           # 문서스토어에서 해당 ID의 문서들 가져오기
           for doc in vectorstore.docstore._dict.values():
               if doc.metadata.get("서비스ID") in doc_ids_to_fetch:
                   all_docs.append(doc)
                   
           if not all_docs:
               print("검색된 문서가 없습니다.")
               return [], user_data, common_ids, info1_only_ids, info2_only_ids
               
       except Exception as e:
           print(f"벡터스토어 로드 또는 문서 검색 중 오류 발생: {e}")
           return [], user_data, common_ids, info1_only_ids, info2_only_ids

       # 최종 문서 목록
       final_docs_ids = []
       
       # 공통 그룹 문서 처리
       if common_ids:  # 공통 문서가 있는 경우에만 처리
           try:
               common_docs = [doc for doc in all_docs if doc.metadata.get("서비스ID") in common_ids]
               if common_docs:
                   reranked_common = rerank_documents_and_extract_service_ids(
                       common_docs,
                       stepback_question,
                       top_k=min(FINAL_DOCS_COUNT, len(common_docs))
                   )
                   if reranked_common:
                       final_docs_ids = reranked_common
           except Exception as e:
               print(f"공통 문서 re-ranking 중 오류 발생: {e}")
               # 오류 발생 시 공통 문서 ID를 그대로 사용
               final_docs_ids = list(common_ids)[:min(FINAL_DOCS_COUNT, len(common_ids))]
       
       current_count = len(final_docs_ids)

       # 공통 문서가 없거나 FINAL_DOCS_COUNT보다 부족하다면, 추가 그룹에서 필요한 문서를 조회
       if current_count < FINAL_DOCS_COUNT:
           needed = FINAL_DOCS_COUNT - current_count
           # info1_only와 info2_only 중에서 필요한 문서 추출
           additional_ids = info1_only_ids + info2_only_ids
           print(f"추가 문서 ID 수: {len(additional_ids)}")
           
           if additional_ids:
               try:
                   docs_additional = [doc for doc in all_docs if doc.metadata.get("서비스ID") in additional_ids]
                   print(f"추가 문서 수: {len(docs_additional)}")
                   
                   if docs_additional:
                       reranked_additional = rerank_documents_and_extract_service_ids(
                           docs_additional,
                           question,
                           top_k=min(needed, len(docs_additional))
                       )
                       if reranked_additional:
                           final_docs_ids.extend(reranked_additional)
               except Exception as e:
                   print(f"추가 문서 re-ranking 중 오류 발생: {e}")
                   # 오류 발생 시 남은 ID를 그대로 사용
                   remaining_ids = additional_ids[:min(needed, len(additional_ids))]
                   final_docs_ids.extend(remaining_ids)

       print("재정렬 완료")
       
       # final_docs_ids에 있는 문서만 필터링
       documents = [doc for doc in all_docs if doc.metadata.get("서비스ID") in final_docs_ids]
       print(f"최종 문서 수: {len(documents)}")

       return documents, user_data, common_ids, info1_only_ids, info2_only_ids
   except Exception as e:
       print(f"supervisor 함수에서 오류 발생: {e}")
       return [], {}, set(), [], []
   
# async def generate_final_answer(question: str, documents: List[Document], user_data: Dict[str, Any]) -> str:
#    """최종 답변 생성"""
#    print(f"generate_final_answer 함수 시작: {len(documents)} 개의 문서로 답변 생성")

#    try:
#        llm = get_llm()

#        # 문서 내용 필터링: 공백만 있는 문서는 제거
#        valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
#        print(f"유효한 문서 수: {len(valid_documents)}")

#        if not valid_documents:
#            print("유효한 문서가 없음")
#            return "검색 결과에 맞는 혜택 정보를 찾지 못했습니다. 다른 질문으로 다시 시도해주세요."
       
#        # 문서 내용을 텍스트로 변환
#        doc_texts = []
#        for i, doc in enumerate(valid_documents, 1):
#            service_id = doc.metadata.get('서비스ID', '알 수 없음')
#            service_name = doc.metadata.get('서비스명', '제목 없음')
#            doc_text = f"문서 {i}:\n서비스ID: {service_id}\n서비스명: {service_name}\n내용: {doc.page_content}\n"
#            doc_texts.append(doc_text)
       
#        combined_docs = "\n\n".join(doc_texts)
#        print(f"결합된 문서 길이: {len(combined_docs)} 자")
       
#        # 최종 답변 프롬프트 생성
#        prompt = get_final_answer_prompt(
#            question=question,
#            documents=combined_docs,
#            user_data=user_data
#        )
       
#        # 최종 답변 생성
#        chain = prompt | llm | StrOutputParser()
#        print("LLM에 요청 시작")
#        final_answer = await chain.ainvoke({})
#        print(f"LLM에서 응답 받음: {len(final_answer)} 자")
#        return final_answer
#    except Exception as e:
#        print(f"generate_final_answer 함수에서 오류 발생: {e}")
#        return f"답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."