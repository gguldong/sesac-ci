from fastapi import APIRouter, HTTPException
from app.services.policy_classifier import classify_policy_question
from app.services.rag_service import retrieve_relevant_policies
from typing import Dict, List, Optional
from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class ClassificationResponse(BaseModel):
    is_policy_question: bool

class PolicyRetrieveResponse(BaseModel):
    policies: List[Dict] = []

router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_message(request: MessageRequest):
    try:
        print(f"/classify 호출완료: {request.message}")
        is_policy = classify_policy_question(request.message)
        return ClassificationResponse(is_policy_question=is_policy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/retrieve", response_model=PolicyRetrieveResponse)
async def retrieve_policies(request: QueryRequest):
    try:
        policies = retrieve_relevant_policies(
            query=request.query, 
            user_id=request.user_id,
        )
        return PolicyRetrieveResponse(policies=policies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy retrieval failed: {str(e)}") 