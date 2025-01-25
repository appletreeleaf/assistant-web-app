from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Core
from langchain import hub
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# Document loaders
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, WebBaseLoader

import bs4

# Embeddings
from langchain.embeddings import OpenAIEmbeddings

# Vector store
from langchain.vectorstores import FAISS

# Text splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Retrievers
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# LangChain tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

# Agents
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Message history
from langchain_community.chat_message_histories import ChatMessageHistory

# Cross encoders
from langchain.retrievers.document_compressors import CrossEncoderReranker
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
async def chat(user_input: str = Form(...), file: UploadFile = File(None)):
    print(f"User input: {user_input}")  # 로그 확인
    # 파일이 업로드된 경우 처리
    if file:
        # 파일 내용을 읽기 (예: 텍스트 파일)
        contents = await file.read()
        # 예시로 파일 내용을 문자열로 변환 (필요에 따라 추가 처리 가능)
        file_content = contents.decode('utf-8')
        user_input += f"\nFile content: {file_content}"

    # LangChain을 통해 챗봇의 응답 생성
    response = chain.invoke(user_input)
    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
