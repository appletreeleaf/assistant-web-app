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

    response = chain.invoke(user_input)
    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
