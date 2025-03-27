import os
from dotenv import load_dotenv

from typing import List, Optional
from fastapi import UploadFile
import bs4

from langchain import hub
# Base chat message history
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def create_upload_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created upload directory: {directory}")

def get_document_loaders(file_name):

    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8"),
    }
    
    for extension, loader in loaders.items():
        if file_name.endswith(extension):
            print(extension[1:], "loader를 return합니다.")
            return loader(file_name)
    
    return None

def get_web_loader(url):
    print(f"get_web_loader's url: {url}")
    loader = WebBaseLoader(
        web_paths=(url, ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
        },
    )
    return loader
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")

def get_documents(loader, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_document = loader.load_and_split(text_splitter=text_splitter)
    return splitted_document

def get_vectorstore(doc_list):
    embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_documents(doc_list, embeddings)
    return vectorstore

from langchain_core.prompts import MessagesPlaceholder
def get_prompt():
    general_prompt = ChatPromptTemplate.from_messages(
        [
            # 시스템 메시지
            ("system", "You are a helpful assistant."),
            # 대화 기록을 위한 Placeholder
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),  # 질문
        ]
    )
    summary_stuff_prompt = hub.pull("teddynote/summary-stuff-documents-korean")
    summary_map_prompt = hub.pull("teddynote/map-prompt")
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")

    prompts = {
        "document_search": agent_prompt,
        "summarize": {"short": summary_stuff_prompt, "long": summary_map_prompt},
        "translation" : agent_prompt,
        "web_search": agent_prompt,
        "general_prompt": general_prompt
    }
    
    return prompts

from langchain.chains.combine_documents import create_stuff_documents_chain

def get_chain(prompt):
    llm = get_llm()
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_chain


def get_agent_executor(usage, retriever):
    """
    Returns the agent executor object.

    Returns:
        agent_executor: The agent executor object.
    """
    search = TavilySearchResults(k=6)
    tool = create_retriever_tool(
        retriever=retriever,
        name="document_search",
        description="use this tool to search information from the PDF document"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent_prompt = get_prompt()[usage]
    agent = create_openai_functions_agent(llm, [search, tool], agent_prompt)
    return AgentExecutor(agent=agent, tools=[search, tool], verbose=False)


def generate_questions(answer) -> List:
    """
    Generate questions from the given text using OpenAI's GPT model.

    Args:
        text: The input text from which to generate questions.

    Returns:
        List of generated questions.
    """
    prompt = PromptTemplate.from_template(
        """
        You are an helpful assistant.
        Based on the given answer, generate follow-up questions. 
        You must 3 questions and length must under 50 characters.

        Answer: 
        {answer}

        Questions:
        """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

    chain = prompt | llm

    response = chain.invoke(answer)
    following_questions = response.content.strip().split("\n")
    return following_questions

def remove_sources(response: str) -> str:
    """
    Removes source links from the agent's response.

    Args:
        response (str): The original response from the agent.

    Returns:
        str: The response without source links.
    """
    # 출처 링크를 제거하는 정규 표현식
    import re
    return re.sub(r'\[출처\]\(.*?\)', '', response)


UPLOAD_DIRECTORY = "uploads"
uploaded_files = {}  # 세션 ID에 따른 업로드된 파일을 저장할 딕셔너리
async def process_documents(doc_list: List, 
                            files: Optional[List[UploadFile]] = None, 
                            url: Optional[str] = None,
                            session_id: str = None) -> List:
    if files:
        for file in files:
            file_name = os.path.join(UPLOAD_DIRECTORY, file.filename)  # 업로드할 디렉토리 지정 
            with open(file_name, "wb") as f:
                f.write(await file.read())
            print(f"Uploaded file: {file.filename}")

            # 파일의 확장자에 따라 적절한 로더 사용
            loader = get_document_loaders(file_name)
            splitted_documents = get_documents(loader=loader, chunk_size=500, chunk_overlap=50)
            doc_list.extend(splitted_documents)
            print(splitted_documents[0])  # 첫 번째 문서 출력
            # 업로드된 파일 목록에 추가

            if session_id not in uploaded_files:
                uploaded_files[session_id] = []
            uploaded_files[session_id].append(file_name)  # 세션 ID에 파일 경로 추가
            os.remove(file_name)

    return doc_list

async def process_url(
                    doc_list: List, 
                    files: Optional[List[UploadFile]] = None, 
                    url: Optional[str] = None,
                    session_id: str = None) -> List:
    if url:
        print(f"Uploaded url: {url}")

        # 파일의 확장자에 따라 적절한 로더 사용
        loader = get_web_loader(url)
        splitted_documents = get_documents(loader=loader, chunk_size=200, chunk_overlap=50)
        doc_list.extend(splitted_documents)
        
        if doc_list:
            print(len(doc_list))
        else:
            print("빈 문서입니다.")
        # 폴더에 url 정보 추가가
        if session_id not in uploaded_files:
            uploaded_files[session_id] = []
        uploaded_files[session_id].append(url)  # 세션 ID에 파일 경로 추가
    return doc_list

session_histories = {}  # 세션 히스토리를 저장할 딕셔너리
# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:  # 세션 ID가 없으면
        session_histories[session_id] = ChatMessageHistory()  # 새로운 ChatMessageHistory 객체 생성
    # print(session_histories.keys())
    return session_histories[session_id]  # 해당 세션 ID에 대한 세션 기록 반환


async def reset_session(session_id: str):
    session_history = get_session_history(session_id)
    session_history.clear()  # 세션 내역 삭제 (예시)   
    print(f"Session ID: {session_history}")
    return {"message": "Session reset successful"}

def validate_url(url):
    if not url.startswith(('http://', 'https://')):
        raise ValueError("유효하지 않은 url입니다")
    return url

