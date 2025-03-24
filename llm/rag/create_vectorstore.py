import os
import json
import time
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config import JSON_FILE_PATH
# None 값이 있는 경우 "정보 없음"으로 변환하는 함수
def clean_metadata(value):
    return value if value is not None else "정보 없음"

# FAQ + 자연어 문장 스타일을 하나의 문서로 결합하는 함수
def generate_selected_text(obj):
    return (
        # FAQ 스타일
        f"Q: 이 서비스의 이름은 무엇인가요?\nA: {clean_metadata(obj.get('서비스명'))}\n\n"
        f"Q: 이 서비스의 ID는 무엇인가요?\nA: {clean_metadata(obj.get('서비스ID'))}\n\n"
        f"Q: 이 서비스는 어떤 기관에서 제공하나요?\nA: {clean_metadata(obj.get('부서명'))}\n\n"
        f"Q: 이 서비스는 어떤 분야에 속하나요?\nA: {clean_metadata(obj.get('서비스분야'))}\n\n"
        f"Q: 이 서비스의 목적은 무엇인가요?\nA: {clean_metadata(obj.get('서비스목적요약'))}\n\n"
        f"Q: 이 서비스를 받을 수 있는 대상은 누구인가요?\nA: {clean_metadata(obj.get('지원대상'))}\n\n"
        f"Q: 지원 내용은 무엇인가요?\nA: {clean_metadata(obj.get('지원내용'))}\n\n"
        f"Q: 이 서비스의 선정 기준은 무엇인가요?\nA: {clean_metadata(obj.get('선정기준'))}\n\n"
        f"Q: 이 서비스의 지원 유형은 무엇인가요?\nA: {clean_metadata(obj.get('지원유형'))}\n\n"
        f"Q: 언제까지 신청할 수 있나요?\nA: {clean_metadata(obj.get('신청기한'))}\n\n"
        f"Q: 신청 방법은 무엇인가요?\nA: {clean_metadata(obj.get('신청방법'))}\n\n"
        f"Q: 어디에서 신청할 수 있나요?\nA: {clean_metadata(obj.get('접수기관'))}\n\n"
        # 자연어 문장 스타일
        f"{clean_metadata(obj.get('서비스명'))} 서비스 (서비스ID: {clean_metadata(obj.get('서비스ID'))})는 "
        f"{clean_metadata(obj.get('부서명'))}에서 운영하는 {clean_metadata(obj.get('서비스분야'))} 분야의 서비스입니다. "
        f"이 서비스는 {clean_metadata(obj.get('서비스목적요약'))}을 목표로 합니다. "
        f"이 서비스를 받을 수 있는 대상은 {clean_metadata(obj.get('지원대상'))}입니다. "
        f"주요 지원 내용은 다음과 같습니다: {clean_metadata(obj.get('지원내용'))}. "
        f"이 서비스의 선정 기준은 다음과 같습니다: {clean_metadata(obj.get('선정기준'))}. "
        f"지원 유형은 {clean_metadata(obj.get('지원유형'))}입니다. "
        f"신청 기한은 {clean_metadata(obj.get('신청기한'))}까지이며, 신청 방법은 {clean_metadata(obj.get('신청방법'))}입니다. "
        f"관련 문의 및 신청은 {clean_metadata(obj.get('접수기관'))}에서 할 수 있습니다."
    )

# 타이머 시작
start_time = time.time()

# JSON 파일 로드
json_file_path = JSON_FILE_PATH
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

if not isinstance(json_data, list):
    raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

# 문서 변환 및 저장
documents = []
for obj in json_data:
    selected_text = generate_selected_text(obj)
    metadata = {
        "서비스ID": clean_metadata(obj.get("서비스ID")),
        "서비스명": clean_metadata(obj.get("서비스명")),
        "서비스목적요약": clean_metadata(obj.get("서비스목적요약")),
        "신청기한": clean_metadata(obj.get("신청기한")),
        "지원내용": clean_metadata(obj.get("지원내용")),
        "서비스분야": clean_metadata(obj.get("서비스분야")),
        "선정기준": clean_metadata(obj.get("선정기준")),
        "지원유형": clean_metadata(obj.get("지원유형")),
    }
    documents.append(Document(page_content=selected_text, metadata=metadata))

print(f"문서 변환 완료: {len(documents)}개 문서 생성됨.")

# HuggingFace 기반 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")

# FAISS 벡터DB 저장 경로
persist_directory = "./dragonkue"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 배치 사이즈 설정 (한 번에 100개 문서씩 처리)
batch_size = 100

if len(documents) == 0:
    raise ValueError("저장할 문서가 없습니다.")

# 첫 배치로 FAISS 인덱스 생성
vectorstore = FAISS.from_documents(documents[0:batch_size], embedding_model)
print(f"첫 배치(0~{batch_size}) 인덱스 생성 완료.")

# 배치 추가 진행 (tqdm으로 진행 상황 및 ETA 표시)
for i in tqdm(range(batch_size, len(documents), batch_size), desc="배치 처리 진행", unit="batch"):
    batch = documents[i:i + batch_size]
    vectorstore.add_documents(batch)

# 인덱스 저장 (persist)
vectorstore.save_local(persist_directory)
print(f"FAISS 벡터DB 저장 완료. 저장 경로: {persist_directory}")

# 타이머 종료 및 소요 시간 출력
end_time = time.time()
elapsed = end_time - start_time
print(f"총 {len(documents)}개의 문서가 FAISS 벡터DB에 저장되었습니다. 소요 시간: {elapsed:.2f}초")
