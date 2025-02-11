import os
from typing import List
from fastapi import Request
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import create_upload_directory, get_document_loaders, get_documents, get_vectorstore, get_chain
from models import UserInput  # Pydantic 모델 가져오기

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
    usage: str = Form(...),  # 선택한 용도를 받는 부분
):
    print(f"User input: {user_input}")
    print(f"Selected usage: {usage}")
    
    doc_list = []
    retriever = None
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
            print(doc_list[0])
            os.remove(file_name)

        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        retriever = vector_store.as_retriever()
    
    chain = get_chain(usage, retriever)
    response = chain.invoke(user_input)
    print(f"response: {response.content}")
    return {"response": response.content}
