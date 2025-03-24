"""
프롬프트 템플릿 정의 모듈
"""
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
from config import TEXT_TO_SQL_INSTRUCTION, BENEFITS_SCHEMA

def get_text_to_sql_prompt(question: str, user_data: Dict[str, Any], samples: List[Dict[str, str]]) -> PromptTemplate:
    """Text-to-SQL 프롬프트 템플릿 생성"""
    template = """
### Instruction: 
You are an SQL bot that generates SQL queries. Use the given Context and Sample Data to generate an SQL query that solves the provided Question. Generate only the SQL query.**\n
### Context:
{schema}
### Question:
{area} {district}에 사는 {age}세 {gender} {personal_category}에 {household_category}이고 혜택 {question}
### SQL: 
### Sample Data: 
[Sample1]
Question: {sample_question_1}
SQL: {sample_sql_1},
[Sample2]
Question: {sample_question_2}
SQL: {sample_sql_2}

위 정보를 바탕으로 benefits 테이블을 조회하는 SQL 쿼리를 생성해주세요.
반드시 SELECT 절에 service_id 필드를 포함해야 합니다.
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=[]
    )
    
    # 프롬프트 입력값 미리 채우기
    prompt = prompt.partial(
        instructions=TEXT_TO_SQL_INSTRUCTION,
        age=user_data.get("age", "알 수 없음"),
        gender=user_data.get("gender", "알 수 없음"),
        schema=BENEFITS_SCHEMA,
        question=question,
        sample_question_1=samples[0]["question"] if len(samples) > 0 else "",
        sample_sql_1=samples[0]["sql"] if len(samples) > 0 else "",
        sample_question_2=samples[1]["question"] if len(samples) > 1 else "",
        sample_sql_2=samples[1]["sql"] if len(samples) > 1 else ""
    )
    
    return prompt

def get_final_answer_prompt(question: str, documents: str, user_data: Dict[str, Any]) -> PromptTemplate:
    """최종 답변 생성 프롬프트 템플릿"""
    template = """
사용자 질문에 대한 답변을 생성해주세요.

## 질문
{question}

## 관련 문서 정보
{documents}

## 사용자 정보
나이: {age}
성별: {gender}

위 정보를 바탕으로 다음 형식에 맞게 답변을 생성해주세요:

1. 질문 요약: 사용자의 질문을 한 줄로 요약합니다.
2. 핵심 답변: 질문에 대한 직접적인 답변을 제공합니다.
3. 상세 정보: 관련 문서에서 찾은 추가 정보를 제공합니다.
4. 맞춤 추천: 사용자 정보를 고려한 맞춤형 추천이나 조언을 제공합니다.

답변은 친절하고 명확하게 작성해주세요.
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=[]
    )
    
    # 프롬프트 입력값 미리 채우기
    prompt = prompt.partial(
        question=question,
        documents=documents,
        age=user_data.get("age", "알 수 없음"),
        gender=user_data.get("gender", "알 수 없음")
    )
    
    return prompt