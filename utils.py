import os
from dotenv import load_dotenv

from typing import List
import bs4

from langchain import hub
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
    loader = WebBaseLoader(
        web_paths=[url],
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

def get_documents(loader, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split(text_splitter=text_splitter)

def get_vectorstore(doc_list):
    embeddings = OpenAIEmbeddings() 
    return faiss.FAISS.from_documents(doc_list, embeddings)

def get_prompt(usage):
    basic_prompt = PromptTemplate(
            template="""You are a helpful assistant. 
            Answer the following question. 
            Answer in Korean: {user_input}""",
            input_variables=["user_input"]
        )
    rag_prompt = hub.pull("rlm/rag-prompt")

    prompts = {
        "general": basic_prompt,
        "document_search": rag_prompt,
    }
    
    return prompts[usage]


def get_chain(usage, retriever):
    prompt = get_prompt(usage)
    if usage == "document_search":
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
    if usage == 'general':
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=1)
    return chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

def get_agent_executor(retriever):
    """
    Returns the agent executor object.

    Returns:
        agent_executor: The agent executor object.
    """
    search = TavilySearchResults(k=3)
    tool = create_retriever_tool(
        retriever=retriever,
        name="search_documents",
        description="Searches and returns relevant excerpts from the uploaded documents."
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, [search, tool], agent_prompt)
    return AgentExecutor(agent=agent, tools=[search, tool], verbose=True)
# Base chat message history
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory



# 세션 ID를 기반으로 세션 기록을 가져오는 함수

def get_session_history(session_id: str, session_histories) -> BaseChatMessageHistory:
    # 해당 세션 ID에 대한 세션 기록 반환
    if session_id not in session_histories:  # 세션 ID가 없으면
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]  # 해당 세션 ID에 대한 세션 기록 반환