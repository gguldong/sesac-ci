"""
RAG 시스템 메인 실행 파일
"""
import asyncio
import logging
from supervisor import run_supervisor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_log.txt", encoding='utf-8'),
        logging.StreamHandler()  # 콘솔 출력도 유지
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"질문 처리 시작: '{question}' (사용자 ID: {user_id})")
    
    try:
        # 문서와 사용자 정보 수집
        documents, user_data, common_ids, vector_only_ids, sql_only_ids = await run_supervisor(question, user_id)
        #final_answer = await generate_final_answer(question, documents, user_data)
        logger.critical(f"검색된 문서: {documents}") 
        documents_dict = {
            f"문서번호 {i+1}": {
            "내용": doc.page_content,
            "메타데이터": doc.metadata
                } for i, doc in enumerate(documents)
        }
  
        return documents_dict, common_ids, vector_only_ids, sql_only_ids
            
    except Exception as e:
            error_msg = f"오류 발생: {e}"
            print(error_msg)
            logger.error(error_msg)
            # 예외 발생 시에도 API가 기대하는 형식과 일치하는 값 반환
            return {}, [], [], []

# 실행 예시
if __name__ == "__main__":
    user_question = "송파구 48세 남자 서비스 혜택 알려줘."
    user_id = "123" 
    
    result = asyncio.run(main(user_question, user_id))
    print("\n최종 답변:")
    logger.info(f"\n최종 답변: {result}")
    print(result)