import os
import logging
from dotenv import load_dotenv
from typing import List, Optional
from fastapi import Request
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import create_upload_directory, reset_session
from chat_logic.agentic_rag_chat import handle_rag  # 모듈 import
from chat_logic.summary_chat import handle_summarize
from chat_logic.report_maker import handle_report
from chat_logic.selres_chat import handle_selres

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
    elif usage=="write_report":
        return await handle_report(user_input, files, session_id, url, usage)
    elif usage=="select_restaurant":
        return await handle_selres(user_input, files, session_id, url, usage)

@router.post("/reset")
async def reset(session_id: str = Form(...)):
    """Reset the session data."""
    # 세션 ID 유효성 체크
    if not session_id or not isinstance(session_id, str) or len(session_id) < 1:
        raise HTTPException(status_code=400, detail="유효한 세션 ID를 입력하세요.")

    return await reset_session(session_id)  # reset_session 함수 호출
