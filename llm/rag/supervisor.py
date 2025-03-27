"""
RAG 시스템의 전체 실행을 조율하는 수퍼바이저 모듈
"""
import os
import asyncio
import time
from typing import List, Dict, Any, Tuple, Set
from huggingface_hub import login
from dotenv import load_dotenv
from langchain.schema import Document

from config import TOP_K_INITIAL, TOP_K_RERANK, FINAL_DOCS_COUNT, VECTORSTORE_A_PATH, JSON_FILE_PATH
from retriever import get_similarity_results, rerank_documents_and_extract_service_ids
from sql_generator import get_sql_results
from database import get_user_data
from vectorstore import load_vectorstore_a
from step_back_rerank import generate_step_back_query, StepBackRAG

async def run_supervisor(question: str, user_id: str) -> Tuple[List[Document], Dict[str, Any], Set, List, List]:
    try:
        t_overall_start = time.time()  # 전체 프로세스 시작 시간 측정
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("HUGGINGFACE_TOKEN이 설정되지 않았습니다. 기본 모델을 사용합니다.")
        if hf_token:
            login(token=hf_token)
 
        print(f"step back 전 질문 : {question}")
        try:
            # RAG 인스턴스 생성
            rag = StepBackRAG(VECTORSTORE_A_PATH, JSON_FILE_PATH)
            stepback_question = generate_step_back_query(question, rag.llm)
            print(f"step back 후 질문 : {stepback_question}")
        except Exception as e:
            print(f"Step back 질문 생성 중 오류 발생: {e}")
            stepback_question = question
        
        # 사용자 정보는 한 번만 조회하여 재사용
        t_user = time.time()
        user_data = await get_user_data(user_id)
        t_user_end = time.time()
        print(f"[타이밍] 사용자 정보 조회: {t_user_end - t_user:.2f} 초")
        
        # 정보1과 정보2를 병렬로 수집 (user_data를 get_sql_results에 전달하여 중복 호출 제거)
        info1_task = get_similarity_results(question, stepback_question)
        info2_task = get_sql_results(stepback_question, user_id, user_data)
        t_parallel = time.time()
        info1, info2 = await asyncio.gather(info1_task, info2_task)
        t_parallel_end = time.time()
        print(f"[타이밍] 정보1, 정보2 수집 (병렬): {t_parallel_end - t_parallel:.2f} 초")

        print(f"유사도 검색: {len(info1)}개 문서 ID, SQL 검색: {len(info2)}개 service_id")
        common_ids = set(info1).intersection(set(info2))
        info1_only_ids = [doc_id for doc_id in info1 if doc_id not in common_ids]
        info2_only_ids = [doc_id for doc_id in info2 if doc_id not in common_ids]
        print(f"공통 문서 ID: {common_ids}")
        print(f"백터검색 전용 문서 ID: {info1_only_ids}")
        #print(f"sql쿼리 전용 문서 ID: {info2_only_ids}")

        # 벡터스토어 로딩 및 타이밍 측정
        t_vectorstore = time.time()
        vectorstore = load_vectorstore_a()
        t_vectorstore_end = time.time()
        print(f"[타이밍] 벡터스토어 로딩: {t_vectorstore_end - t_vectorstore:.2f} 초")
        
        doc_ids_to_fetch = list(common_ids) + info1_only_ids + info2_only_ids
        doc_ids_to_fetch = [doc_id for doc_id in doc_ids_to_fetch if doc_id]
        if not doc_ids_to_fetch:
            print("검색된 문서 ID가 없습니다.")
            return [], user_data, common_ids, info1_only_ids, info2_only_ids
           
        all_docs = []
        for doc in vectorstore.docstore._dict.values():
            if doc.metadata.get("서비스ID") in doc_ids_to_fetch:
                all_docs.append(doc)
               
        if not all_docs:
            print("검색된 문서가 없습니다.")
            return [], user_data, common_ids, info1_only_ids, info2_only_ids

        final_docs_ids = []
       
        # 공통 그룹 문서 처리 (타이밍 측정 포함)
        if common_ids:
            try:
                common_docs = [doc for doc in all_docs if doc.metadata.get("서비스ID") in common_ids]
                t_common = time.time()
                if common_docs:
                    reranked_common = rerank_documents_and_extract_service_ids(
                        common_docs,
                        stepback_question,
                        top_k=min(FINAL_DOCS_COUNT, len(common_docs))
                    )
                    t_common_end = time.time()
                    print(f"[타이밍] 공통 문서 re-ranking: {t_common_end - t_common:.2f} 초")
                    if reranked_common:
                        final_docs_ids = reranked_common
                else:
                    print("공통 문서가 없습니다.")
            except Exception as e:
                print(f"공통 문서 re-ranking 중 오류 발생: {e}")
                final_docs_ids = list(common_ids)[:min(FINAL_DOCS_COUNT, len(common_ids))]
       
        current_count = len(final_docs_ids)
        if current_count < FINAL_DOCS_COUNT:
            needed = FINAL_DOCS_COUNT - current_count
            additional_ids = info1_only_ids
            print(f"추가 문서 ID 수: {len(additional_ids)}")
            if additional_ids:
                try:
                    docs_additional = [doc for doc in all_docs if doc.metadata.get("서비스ID") in additional_ids]
                    print(f"추가 문서 수: {len(docs_additional)}")
                    if docs_additional:
                        t_additional = time.time()
                        reranked_additional = rerank_documents_and_extract_service_ids(
                            docs_additional,
                            question,
                            top_k=min(needed, len(docs_additional))
                        )
                        t_additional_end = time.time()
                        print(f"[타이밍] 추가 문서 re-ranking: {t_additional_end - t_additional:.2f} 초")
                        if reranked_additional:
                            final_docs_ids.extend(reranked_additional)
                    else:
                        print("추가 문서가 없습니다.")
                except Exception as e:
                    print(f"추가 문서 re-ranking 중 오류 발생: {e}")
                    remaining_ids = additional_ids[:min(needed, len(additional_ids))]
                    final_docs_ids.extend(remaining_ids)

        print("재정렬 완료")
        documents = [doc for doc in all_docs if doc.metadata.get("서비스ID") in final_docs_ids]
        print(f"최종 문서 수: {len(documents)}")
        
        t_overall_end = time.time()
        print(f"[타이밍] 전체 프로세스 소요 시간: {t_overall_end - t_overall_start:.2f} 초")
        return documents, user_data, common_ids, info1_only_ids, info2_only_ids
    except Exception as e:
        print(f"run_supervisor 함수에서 오류 발생: {e}")
        return [], {}, set(), [], []