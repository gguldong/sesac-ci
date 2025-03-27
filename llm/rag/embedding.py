"""
임베딩 모델 및 벡터스토어 관리 기능을 제공하는 모듈
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL

# 전역 변수로 임베딩 모델 캐싱
_embedding_model = None

def get_embedding_model():
    """임베딩 모델을 싱글톤 패턴으로 로드"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model

# 질문 벡터스토어 경로 및 전역 변수
QUESTION_VECTORSTORE_PATH = './question_embedded'
_question_vectorstore = None

def get_question_vectorstore():
    """질문 벡터스토어를 싱글톤 패턴으로 로드"""
    global _question_vectorstore
    if _question_vectorstore is None:
        try:
            # 보안 관련 파라미터 추가
            embedding_model = get_embedding_model()
            _question_vectorstore = FAISS.load_local(
                QUESTION_VECTORSTORE_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True  # 신뢰할 수 있는 소스에서만 True로 설정
            )
            print(f"질문 벡터스토어 로드 완료: {QUESTION_VECTORSTORE_PATH}")
        except Exception as e:
            print(f"질문 벡터스토어 로드 실패: {e}")
            # 실패 시 처리 - 예외 처리 개선
            print("대체 로직 사용 중...")
            # 빈 벡터스토어 대신, 기본 예제를 사용하는 방식으로 변경
            return None  # 또는 기본 예제를 반환하는 함수 호출
    return _question_vectorstore