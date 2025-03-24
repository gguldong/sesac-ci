"""
RAG 시스템 메인 실행 파일
"""
import asyncio
from processors import supervisor

async def main(question: str, user_id: str) -> tuple:
    """
    RAG 시스템 메인 함수
    
    Args:
        question: 사용자 질문
        user_id: 사용자 ID
        
    Returns:
        documents_dict: 문서 딕셔너리
        common_ids: 공통 ID 목록
        vector_only_ids: 벡터 검색에서만 찾은 ID 목록
        sql_only_ids: SQL 검색에서만 찾은 ID 목록
    """
    print(f"질문 처리 시작: '{question}' (사용자 ID: {user_id})")
    
    try:
        # 문서와 사용자 정보 수집
        # supervisor 함수 호출 부분 수정
        documents, user_data, common_ids, info1_only_ids, info2_only_ids = await supervisor(question, user_id)
        #final_answer = await generate_final_answer(question, documents, user_data)

        documents_dict = {
            f"문서번호 {i+1}": {
            "내용": doc.page_content,
            "메타데이터": doc.metadata
                } for i, doc in enumerate(documents)
        }
        vector_only_ids = info1_only_ids
        sql_only_ids = info2_only_ids
        return documents_dict, common_ids, vector_only_ids, sql_only_ids
            
    except Exception as e:
        print(f"오류 발생: {e}")
        # 예외 발생 시에도 API가 기대하는 형식과 일치하는 값 반환
        return {}, [], [], []

# 실행 예시
if __name__ == "__main__":
    user_question = "충청남도 논산시에 거주하는 32세 남자이며 예비부부/난임인 사람에게 적합한 지원을 알려줘."
    user_id = "123" 
    
    result = asyncio.run(main(user_question, user_id))
    print("\n최종 답변:")
    print(result)