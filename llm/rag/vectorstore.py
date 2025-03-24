"""
벡터스토어 관련 기능을 제공하는 모듈
"""
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from typing import List
from models import load_embedding_model
from config import VECTORSTORE_A_PATH

# 경로 및 초기 변수 설정
# _embedding_model은 실제 모델 객체가 저장될 변수이므로 초기값을 None으로 설정합니다.
_embedding_model = None  
# documents는 JSON 파일 경로가 아니라, 문서 객체 리스트여야 합니다.
# 아래 예시는 JSON 파일을 로드해서 Document 객체 리스트로 변환하는 부분(구현 예시)입니다.
import json


def load_documents(json_path: str) -> List[Document]:
    """JSON 파일을 로드하여 Document 객체 리스트로 변환 (예시)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data의 구조에 따라 Document 객체를 생성합니다.
    # 예를 들어 data가 리스트 형식이고, 각 항목에 'text' 필드가 있다고 가정합니다.
    documents = [Document(page_content=item.get("text", ""), metadata=item) for item in data]
    return documents

documents = load_documents("20250304.json")

# 전역 변수로 벡터스토어 캐싱
_vectorstore_a = None
# _vectorstore_c = None

def get_embedding_model():
    """임베딩 모델 싱글톤 패턴으로 로드"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model()
    return _embedding_model

def load_vectorstore_a():
    """documents1 벡터스토어 로드"""
    global _vectorstore_a
    embedding_model = get_embedding_model()
    try:
        _vectorstore_a = FAISS.load_local(VECTORSTORE_A_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("기존 FAISS 벡터스토어 로드 성공.")
    except Exception as e:
        print("FAISS 인덱스가 존재하지 않음. 새로 생성합니다...")
        try:
            _vectorstore_a = FAISS.from_documents(documents, embedding_model)
            _vectorstore_a.save_local(VECTORSTORE_A_PATH)
            print(f"총 {len(documents)}개의 문서가 FAISS 벡터 DB에 저장됨.")
        except Exception as e:
            print(f"FAISS 생성 중 오류 발생: {e}")
            raise
    return _vectorstore_a

# def load_vectorstore_c():
#     """text-to-sql 학습데이터 벡터스토어 로드"""
#     global _vectorstore_c
#     if _vectorstore_c is None:
#         embedding = get_embedding_model()
#         _vectorstore_c = FAISS.load_local(VECTORSTORE_C_PATH, embedding, allow_dangerous_deserialization=True)
#     return _vectorstore_c

def retrieve_similar_docs(question: str, store, top_k: int = 30) -> List[Document]:
    """벡터스토어에서 유사 문서 검색"""
    return store.similarity_search(question, k=top_k)

def create_ensemble_retriever(documents: List[Document]) -> EnsembleRetriever:
    """문서 리스트로부터 앙상블 리트리버 생성"""
    if len(documents) < 2:
        # 문서가 충분하지 않으면 기본 FAISS 리트리버 반환
        embedding = get_embedding_model()
        faiss_retriever = FAISS.from_documents(documents, embedding).as_retriever(
            search_kwargs={"k": len(documents)}
        )
        return faiss_retriever
    
    # BM25 리트리버 생성
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = min(2, len(documents))
    
    # FAISS 리트리버 생성
    embedding = get_embedding_model()
    faiss_vectorstore = FAISS.from_documents(documents, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever(
        search_kwargs={"k": min(2, len(documents))}
    )
    
    # 앙상블 리트리버 생성
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

