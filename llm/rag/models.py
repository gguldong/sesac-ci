"""
모델 로드 및 관리를 위한 모듈
"""
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_openai import ChatOpenAI
from config import (
    EMBEDDING_MODEL, 
    EMBEDDING_KWARGS, 
    LLM_MODEL, 
    LLM_TEMPERATURE
)
import torch

HUGGINGFACE_TOKEN = "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"

def load_embedding_model():
    """한국어 임베딩 모델을 로드"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs=EMBEDDING_KWARGS
    )

def get_llm(model_name=LLM_MODEL, temperature=LLM_TEMPERATURE):
    """LLM 모델 인스턴스 생성"""
    return ChatOpenAI(model_name=model_name, temperature=temperature)



from peft import PeftModel

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import Optional, List

# 임시 대안: 더 간단한 LLM 래퍼 구현
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

def load_tunned_model():
    try:
        # 기본 모델 로드 (GPU 사용 및 offload_dir 지정)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.3",
            device_map="auto",
            torch_dtype=torch.float16,
            offload_folder="./offload_dir"  # 오프로딩할 폴더 경로 (폴더가 존재하는지 확인)
        )
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, "./tuned_model")
        
        # 파이프라인 생성 (GPU device 지정)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            temperature=0,
        )
        
        from langchain_huggingface import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        print("파인튜닝 모델 로드 성공")
        return llm
    except Exception as e:
        print("파인튜닝 모델 로드 실패:", e)
        return get_llm()
    