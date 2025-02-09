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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import faiss

# LangChain OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

def get_vectorstore(doc_list):
    """
    Stores document embeddings in a vector store and returns it.

    Args:
        doc_list: The list of documents.

    Returns:
        vectorstore: The vector store containing document embeddings.
    """
    embeddings = OpenAIEmbeddings() #BPE encoding
    return faiss.FAISS.from_documents(doc_list, embeddings)

def get_chain(usage, retriever):
    if usage == "document_search":

        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Answer in Korean.

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )
    else:
        prompt = PromptTemplate(
        template="""You are a helpful assistant. 
        Answer the following question. 
        Answer in Korean: {user_input}""",
        input_variables=["user_input"]
        )
        chain = prompt | llm 
    return chain
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
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)


# Pydantic 모델 정의
class UserInput(BaseModel):
    user_input: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi import Form  # 추가 임포트

@app.post("/chat")
async def chat(
    user_input: str = Form(...), 
    files: List[UploadFile] = File(default=None),
    usage: str = Form(...),  # 선택한 용도를 받는 부분
    ):
    print(f"User input: {user_input}")  # log
    print(f"Selected usage: {usage}")
    # RAG
    doc_list = []
    retriever = None
    if files:
        for file in files:
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            print(f"Uploaded file: {file.filename}")

            # 파일의 확장자에 따라 적절한 로더 사용
            loader = get_document_loaders(file.filename)
            splitted_documents = get_documents(loader=loader, chunk_size=1000, chunk_overlap=100)
            doc_list.extend(splitted_documents)
            print(doc_list[0])
            os.remove(file_path)

        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        retriever = vector_store.as_retriever()
    
    chain = get_chain(usage, retriever)
    response = chain.invoke(user_input)
    print(f"response: {response.content}")
    return {"response": response.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
