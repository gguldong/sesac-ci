"""
사용자 데이터 처리와 관련된 기능을 제공하는 모듈
"""
import datetime
import re
import traceback
from typing import Dict, Any, List

async def is_self_check(user_data: Dict[str, Any], query_data: Dict[str, Any]) -> bool:
    """
    사용자 데이터와 쿼리 데이터를 비교하여 같은 사람인지 확인합니다.
    
    Args:
        user_data: 현재 로그인한 사용자의 데이터
        query_data: 쿼리에서 비교하려는 데이터
        
    Returns:
        bool: 같은 사람으로 판단되면 True, 아니면 False
    """
    # 필수 필드가 모두 있는지 확인
    if not user_data or not query_data:
        return False
    
    # 성별은 반드시 일치해야 함
    if "gender" in query_data:
        if query_data.get("gender") != user_data.get("gender"):
            return False
    
    # 일치하는 필드 수 계산
    matching_fields = 0
    total_fields = 0
    
    # 비교할 중요 필드 목록
    fields_to_check = ["age", "district"]
    
    for field in fields_to_check:
        if field in query_data:
            total_fields += 1
            
            # 나이/생일 비교 로직 (특별 처리)
            if field == "age" and "birthDate" in user_data:
                # 사용자 데이터에서 나이 계산
                try:
                    birthdate_val = user_data.get("birthDate")
                    
                    if isinstance(birthdate_val, str):
                        birthdate = datetime.datetime.strptime(birthdate_val, "%Y-%m-%d").date()
                    elif isinstance(birthdate_val, datetime.date):
                        birthdate = birthdate_val
                    elif isinstance(birthdate_val, datetime.datetime):
                        birthdate = birthdate_val.date()
                    else:
                        continue
                    
                    today = datetime.date.today()
                    calculated_age = today.year - birthdate.year
                    if (today.month, today.day) < (birthdate.month, birthdate.day):
                        calculated_age -= 1
                    
                    query_age = int(query_data.get("age"))
                    
                    # 나이가 같거나 1살 차이 이내면 일치로 간주
                    if abs(calculated_age - query_age) <= 1:
                        matching_fields += 1
                except:
                    pass
                
            # 지역 비교 (부분 일치 허용)
            elif field == "district" and field in user_data:
                user_district = str(user_data.get("district")).lower()
                query_district = str(query_data.get("district")).lower()
                
                # 부분 일치 확인 (예: '서울시 강남구'와 '강남구'도 매칭)
                if user_district in query_district or query_district in user_district:
                    matching_fields += 1
            
            # 일반 필드 비교
            elif field in user_data:
                if str(user_data.get(field)).lower() == str(query_data.get(field)).lower():
                    matching_fields += 1
    
    # 판단 기준: 
    # 1. 성별은 이미 확인했음
    # 2. 필드가 2개 이상이면 절반 이상 일치해야 함
    # 3. 필드가 1개면 반드시 일치해야 함
    if total_fields >= 2:
        return matching_fields >= total_fields / 2
    elif total_fields == 1:
        return matching_fields == 1
    else:
        return False  # 비교할 필드가 없으면 False

async def combine_user_data(sql_query: str, user_data: Dict[str, Any], is_for_self: bool = True, 
                           merge_fields: List[str] = None) -> str:
    """
    사용자 데이터와 SQL 쿼리를 병합합니다.
    
    Args:
        sql_query: 원본 SQL 쿼리
        user_data: 사용자 데이터 딕셔너리
        is_for_self: 본인 조회인지 여부 (True인 경우에만 정보 합침)
        merge_fields: 병합할 특정 필드 리스트 (None이면 모든 가능한 필드 병합)
    
    Returns:
        병합된 SQL 쿼리
    """
    # 필수 조건 확인
    if not sql_query or not user_data:
        return sql_query
    
    # # 본인 조회가 아닌 경우 원래 쿼리 그대로 반환
    # if not is_for_self:
    #     print("본인에 대한 질문이 아님")
    #     return sql_query
    
    try:
        print("본인에 대한 질문임")
        # 병합할 필드가 명시되지 않았다면 기본 필드 설정
        if merge_fields is None:
            merge_fields = ["age", "district"]  # 기본적으로 병합할 필드
        
        additional_conditions = {}
        
        # 나이 정보 계산 (age 필드가 병합 대상일 때만)
        if "age" in merge_fields and "birthDate" in user_data:
            birthdate_val = user_data.get("birthDate")
            
            if birthdate_val:
                try:
                    # 다양한 형태의 birthdate 값 처리
                    if isinstance(birthdate_val, str):
                        # 문자열 형식에 따라 다양한 파싱 시도
                        try:
                            birthdate = datetime.datetime.strptime(birthdate_val, "%Y-%m-%d").date()
                        except ValueError:
                            try:
                                birthdate = datetime.datetime.strptime(birthdate_val, "%Y/%m/%d").date()
                            except ValueError:
                                # 더 많은 형식 추가 가능
                                return sql_query
                    elif isinstance(birthdate_val, datetime.date):
                        birthdate = birthdate_val
                    elif isinstance(birthdate_val, datetime.datetime):
                        birthdate = birthdate_val.date()
                    else:
                        # 지원하지 않는 타입
                        return sql_query
                    
                    today = datetime.date.today()
                    age = today.year - birthdate.year
                    if (today.month, today.day) < (birthdate.month, birthdate.day):
                        age -= 1
                    
                    # 쿼리에 나이 관련 조건이 이미 있는지 확인
                    has_explicit_age_condition = any(re.search(rf"\b{field}\b", sql_query, flags=re.IGNORECASE) 
                                                  for field in ["age", "min_age", "max_age"])
                    
                    # 나이 조건이 없는 경우에만 추가
                    if not has_explicit_age_condition:
                        additional_conditions["min_age"] = f"{age}"
                        additional_conditions["max_age"] = f"{age}"
                except Exception as e:
                    print(f"나이 계산 중 오류 발생: {e}")
        
        # 지역 정보 추가 (district 필드가 병합 대상일 때만)
        if "district" in merge_fields and "district" in user_data:
            district_val = user_data.get("district")
            if district_val and not re.search(r"\bdistrict\b", sql_query, flags=re.IGNORECASE):
                additional_conditions["district"] = f"'{district_val}'"
        
        # 기타 사용자 지정 필드 추가
        for field in merge_fields:
            if field not in ["age", "district"] and field in user_data:
                field_val = user_data.get(field)
                if field_val and not re.search(rf"\b{field}\b", sql_query, flags=re.IGNORECASE):
                    if isinstance(field_val, (int, float)):
                        additional_conditions[field] = f"{field_val}"
                    else:
                        additional_conditions[field] = f"'{field_val}'"
        
        # 추가할 조건이 없으면 원래 쿼리 그대로 반환
        if not additional_conditions:
            return sql_query
            
        # 추가 조건 문자열 구성
        condition = ""
        for column, value in additional_conditions.items():
            if column == "min_age":
                condition += f" AND {column} <= {value}"
            elif column == "max_age":
                condition += f" AND {column} >= {value}"
            else:
                condition += f" AND {column} = {value}"
        
        # SQL 쿼리 병합
        if "WHERE" not in sql_query.upper() and condition:
            # WHERE 절 삽입 위치 찾기
            from_pos = sql_query.upper().find("FROM")
            if from_pos < 0:  # FROM 절이 없으면 원본 쿼리 반환
                return sql_query
                
            from_clause_end = from_pos + 4
            
            # FROM 다음에 나오는 첫 번째 절 찾기
            next_clause_pos = float('inf')
            for keyword in ["ORDER BY", "GROUP BY", "HAVING", "LIMIT"]:
                pos = sql_query.upper().find(keyword, from_clause_end)
                if pos > 0 and pos < next_clause_pos:
                    next_clause_pos = pos
            
            if next_clause_pos < float('inf'):
                combined_query = f"{sql_query[:next_clause_pos]} WHERE 1=1{condition} {sql_query[next_clause_pos:]}"
            else:
                combined_query = f"{sql_query.rstrip(';')} WHERE 1=1{condition};"
        else:
            # WHERE 절이 이미 있는 경우
            # 조건을 삽입할 위치 찾기
            insert_pos = -1
            for keyword in ["GROUP BY", "HAVING", "ORDER BY", "LIMIT"]:
                pos = sql_query.upper().find(keyword)
                if pos > 0:
                    if insert_pos == -1 or pos < insert_pos:
                        insert_pos = pos
            
            if insert_pos > 0:
                combined_query = f"{sql_query[:insert_pos]}{condition} {sql_query[insert_pos:]}"
            else:
                # 끝에 조건 추가
                combined_query = f"{sql_query.rstrip(';')}{condition};"
        
        print(f"고객정보랑 합쳐서 sql 출력: {combined_query}")
        return combined_query
    except Exception as e:
        print(f"combine_user_data 에러 발생: {e}", traceback.format_exc())
        return sql_query  # 에러 발생 시 원래 쿼리 반환