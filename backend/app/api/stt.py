import io
import logging
from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.openai_stt import whisper_stt

router = APIRouter()

# 허용된 오디오 파일 타입
ALLOWED_AUDIO_TYPES = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/webm"]
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

@router.post("/upload-audio/")
async def upload_audio(audio: UploadFile = File(...)):
    # 파일 타입 검증
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_AUDIO_TYPES)}"
        )
    
    logging.info(f"오디오 파일 업로드: {audio.filename}, 타입: {audio.content_type}")
    
    try:
        # 파일 읽기
        contents = await audio.read()
        
        # 파일 크기 검증
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"파일 크기가 너무 큽니다. 최대 허용 크기: {MAX_FILE_SIZE/(1024*1024)}MB"
            )
        
        # 바이트 데이터를 파일 객체로 변환
        audio_bytes = io.BytesIO(contents)
        audio_bytes.name = audio.filename  # 파일명 설정 (일부 API에서 필요)
        
        # STT 변환
        transcription = whisper_stt(audio_bytes)
        
        return {"text": transcription, "filename": audio.filename}
    
    except Exception as e:
        logging.error(f"STT 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="오디오 처리 중 오류가 발생했습니다. 자세한 내용은 로그를 확인하세요."
        )