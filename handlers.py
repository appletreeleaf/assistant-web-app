import os
import logging
from dotenv import load_dotenv
from typing import List, Optional
from fastapi import Request
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import create_upload_directory
from chat_logic.agentic_rag_chat import handle_rag, reset_session  # 모듈 import
from chat_logic.summary_chat import handle_summarize

# 환경변수
load_dotenv()
router = APIRouter()

templates = Jinja2Templates(directory="templates")
UPLOAD_DIRECTORY = "uploads"

@router.on_event("startup")
async def startup_event():
    create_upload_directory(UPLOAD_DIRECTORY)  # 업로드 디렉토리 생성

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/chat")
async def chat(
    user_input: str = Form(...), 
    files: List[UploadFile] = File(default=None),
    session_id: str = Form(...),  # 클라이언트에서 전달받는 세션 ID
    url: Optional[str] = Form(default=None),  # URL 입력을 받는 부분
    usage: str = Form(...),  # 선택한 용도를 받는 부분
):
    
    if usage=="document_search":
        return await handle_rag(user_input, files, session_id, url, usage)
    elif usage=="summarize":
        return await handle_summarize(user_input, files, session_id, url, usage)

@router.post("/reset")
async def reset(session_id: str = Form(...)):
    """Reset the session data."""
    return await reset_session(session_id)  # reset_session 함수 호출
