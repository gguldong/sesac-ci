"""
벡터스토어 관리를 위한 모듈
"""
from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_A_PATH
from embedding import get_embedding_model

# 전역 변수로 벡터스토어 캐싱
_vectorstore_a = None

def load_vectorstore_a():
    """벡터스토어 A를 싱글톤 패턴으로 로드"""
    global _vectorstore_a
    if _vectorstore_a is None:
        embedding_model = get_embedding_model()
        _vectorstore_a = FAISS.load_local(VECTORSTORE_A_PATH, embedding_model, allow_dangerous_deserialization=True)
    return _vectorstore_a