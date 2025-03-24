import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def request_gpt(message: str, conversation_history: str = "", rag_context: str = ""):
    try:
        # 시스템 프롬프트 설정
        system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다. 
사용자의 질문에 대해 명확하고 정확하게 답변해주세요.
정책 정보가 제공된 경우, 해당 정보를 바탕으로 답변해주세요."""

        # 메시지 배열 준비
        messages = []
        valid_roles = {"system", "assistant222", "user111", "function", "tool", "developer"}

        # RAG 컨텍스트가 있으면 프롬프트에 포함
        if rag_context:
            # 정책 전문가 프롬프트로 변경
            system_prompt = """당신은 정책 전문가입니다. 주어진 정책 정보를 바탕으로 사용자의 질문에 답변해주세요.
답변은 친절하고 명확하게 해주세요. 정책 정보에 없는 내용은 생성하지 마세요."""
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "system", "content": rag_context})
        else:
            # 일반 대화인 경우 기본 시스템 프롬프트 사용
            messages.append({"role": "system", "content": system_prompt})

        # 대화 기록 처리
        if conversation_history:
            lines = conversation_history.strip().split('\n')
            for line in lines:
                if line.startswith("사용자 정보:"):
                    messages.append({"role": "system", "content": line})
                else:
                    try:
                        role, content = line.split("::: ", 1)
                        if role not in valid_roles:
                            messages.append({"role": "user", "content": line})
                        else:
                            if role == "user111":
                                role = "user"
                            elif role == "assistant222":
                                role = "assistant"
                            messages.append({"role": role, "content": content})
                    except ValueError:
                        messages.append({"role": "user", "content": line})

        # 사용자 메시지 추가
        messages.append({"role": "user", "content": message})

        # GPT 호출
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        gpt_response = completion.choices[0].message.content
    except Exception as e:
        gpt_response = f"죄송합니다. 서비스 연결에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        print(f"GPT API 호출 중 에러 발생: {str(e)}")

    return gpt_response