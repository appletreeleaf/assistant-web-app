import os
from typing import List, Optional
from fastapi import Request
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.chat_message_histories import ChatMessageHistory
from utils import (create_upload_directory, get_document_loaders, get_documents, 
                   get_vectorstore, get_chain, get_web_loader, get_agent_executor)
import uuid

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIRECTORY = "uploads"
session_histories = {}  # 세션 히스토리를 저장할 딕셔너리
uploaded_files = {}  # 세션 ID에 따른 업로드된 파일을 저장할 딕셔너리

@router.on_event("startup")
async def startup_event():
    create_upload_directory(UPLOAD_DIRECTORY)  # 업로드 디렉토리 생성
    

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def process_documents(doc_list: List, files: Optional[List[UploadFile]] = None, url: Optional[str] = None) -> List:
    if files:
        for file in files:
            file_name = os.path.join(UPLOAD_DIRECTORY, file.filename)  # 업로드할 디렉토리 지정 
            with open(file_name, "wb") as f:
                f.write(await file.read())
            print(f"Uploaded file: {file.filename}")

            # 파일의 확장자에 따라 적절한 로더 사용
            loader = get_document_loaders(file_name)
            splitted_documents = get_documents(loader=loader, chunk_size=1000, chunk_overlap=100)
            doc_list.extend(splitted_documents)
            print(splitted_documents[0])  # 첫 번째 문서 출력
            os.remove(file_name)
    return doc_list

from langchain_core.runnables.history import RunnableWithMessageHistory

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:  # 세션 ID가 없으면
        session_histories[session_id] = ChatMessageHistory()  # 새로운 ChatMessageHistory 객체 생성
    return session_histories[session_id]  # 해당 세션 ID에 대한 세션 기록 반환

@router.post("/chat")
async def chat(
    user_input: str = Form(...), 
    files: List[UploadFile] = File(default=None),
    session_id: str = Form(...),  # 클라이언트에서 전달받는 세션 ID
    url: Optional[str] = Form(default=None),  # URL 입력을 받는 부분
    usage: str = Form(...),  # 선택한 용도를 받는 부분
):
    print(f"Generated session ID: {session_id}")
    print(f"User input: {user_input}")
    print(f"Selected usage: {usage}")
    

    doc_list = []
    retriever = None
    
    if files:
        doc_list = await process_documents(doc_list, files=files, url=url)

        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        
        retriever = vector_store.as_retriever()

    agent_executor = get_agent_executor(retriever)

    # Create agent with chat history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,  # 직접 get_session_history 함수 전달
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # 대화 내역에 사용자 입력 추가
    session_history = get_session_history(session_id)
    session_history.add_message({"role": "user", "content": user_input})

    # 에이전트를 통해 응답 생성
    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}  # 세션 ID를 포함하여 호출
    )

    # 응답을 대화 내역에 추가
    session_history.add_message({"role": "assistant", "content": response["output"]})
    print(history for history in session_history)
    return {"response": response["output"]}

@router.post("/reset")
async def reset(session_id: str = Form(...)):
    """Reset the session data."""
    session_histories.pop(session_id, None)  # 세션 히스토리 초기화
    uploaded_files.pop(session_id, None)  # 업로드한 파일 초기화
    return {"message": "Session reset successful"}
