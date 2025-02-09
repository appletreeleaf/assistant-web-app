from typing import List

# FastAPI
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import faiss

# LangChain OpenAI
from langchain_openai import ChatOpenAI
# LanChain Community
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
import os

def get_document_loaders(file_name):
    """
    Returns the appropriate loader based on the document format.

    Args:
        file_name: The name of the file including the extension.
    
    Returns:
        loader: The corresponding document loader.
    """ 
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8")
    }
    
    for extension, loader in loaders.items():
        if file_name.endswith(extension):
            return loader(file_name)
    
    return None

def get_documents(loader, chunk_size, chunk_overlap):
    """
    Returns the split documents.
    
    Args:
        loader: Document loader.
        chunk_size: Size of the chunks.
        chunk_overlap: Overlap between chunks.

    Returns:
        splitted_documents: The list of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split(text_splitter=text_splitter)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (실제 배포 시에는 특정 출처만 허용하는 것이 좋습니다)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# LangChain ChatGPT 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# 프롬프트 템플릿 정의
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful assistant. Answer the following question: {user_input}"
)

chain = prompt | llm

# Pydantic 모델 정의
class UserInput(BaseModel):
    user_input: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi import Form  # 추가 임포트

@app.post("/chat")
async def chat(user_input: str = Form(...), files: List[UploadFile] = File(default=None)):
    print(f"User input: {user_input}")
        # 파일 처리 (예: 파일 저장)
    doc_list = []
    if files:
        for file in files:
            # 파일의 확장자에 따라 적절한 로더 사용
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            print(f"Uploaded file: {file.filename}")
            loader = get_document_loaders(file.filename)
            splitted_documents = get_documents(loader=loader, chunk_size=1000, chunk_overlap=100)
            doc_list.extend(splitted_documents)
            print(doc_list[1])
            os.remove(file_path)

    response = chain.invoke(user_input)
    print(f"response: {response.content}")
    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
