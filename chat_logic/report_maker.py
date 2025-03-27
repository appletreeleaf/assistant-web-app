from typing import List, Optional
from fastapi import UploadFile
from fastapi.responses import FileResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import (get_vectorstore, get_agent_executor, 
                   generate_questions, remove_sources, process_documents,
                   get_session_history)
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate


from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from utils import get_session_history

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import tool

from langchain_community.agent_toolkits import FileManagementToolkit

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# API 키 정보 로드
load_dotenv()
async def handle_report(user_input: str, files: List[UploadFile], 
                      session_id: str, url: Optional[str], 
                      usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"Selected usage: {usage}")
    
    doc_list = []
    # 파일 처리
    if files:
        doc_list = await process_documents(doc_list, files, session_id)

    if doc_list:
        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        
        retriever = vector_store.as_retriever()
    else:
        retriever = None
    search = TavilySearchResults(k=6)

    # 문서의 내용을 표시하는 템플릿을 정의합니다.
    document_prompt = PromptTemplate.from_template(
        "<document><content>{page_content}</content><page>{page}</page><filename>{source}</filename></document>"
    )

    # retriever 를 도구로 정의합니다.
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="use this tool to search for information in the PDF file",
        document_prompt=document_prompt,
    )


    # DallE API Wrapper를 생성합니다.
    dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)


    # DallE API Wrapper를 도구로 정의합니다.
    @tool
    def dalle_tool(query):
        """use this tool to generate image from text"""
        return dalle.run(query)


    # 작업 디렉토리 경로 설정
    working_directory = "tmp"

    # 파일 관리 도구 생성(파일 쓰기, 읽기, 디렉토리 목록 조회)
    file_tools = FileManagementToolkit(
        root_dir=str(working_directory),
        selected_tools=["write_file", "read_file", "list_directory"],
    ).get_tools()

    # 생성된 파일 관리 도구 출력
    file_tools

    tools = file_tools + [
        retriever_tool,
        search,
        dalle_tool,
    ]

    # 최종 도구 목록 출력
    tools


    # session_id 를 저장할 딕셔너리 생성
    store = {}

    # 프롬프트 생성
    # 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "You are a professional researcher. "
                "You can use the pdf_search tool to search for information in the PDF file. "
                "You can find further information by using search tool. "
                "You can use image generation tool to generate image from text. "
                "Finally, you can use file management tool to save your research result into files.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    # LLM 생성
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
    )


    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    # 에이전트 실행
    result = agent_with_chat_history.stream(
        {
            "input": "삼성전자가 개발한 `생성형 AI` 와 관련된 유용한 정보들을 PDF 문서에서 찾아서 bullet point로 정리해 주세요. "
            "한글로 작성해주세요."
            "다음으로는 `report.md` 파일을 새롭게 생성하여 정리한 내용을 저장해주세요. \n\n"
            "#작성방법: \n"
            "1. markdown header 2 크기로 적절한 제목을 작성하세요. \n"
            "2. 발췌한 PDF 문서의 페이지 번호, 파일명을 기입하세요(예시: page 10, filename.pdf). \n"
            "3. 정리된 bullet point를 작성하세요. \n"
            "4. 작성이 완료되면 파일을 `report.md` 에 저장하세요. \n"
            "5. 마지막으로 저장한 `report.md` 파일을 읽어서 출력해 주세요. \n"
        },
        config={"configurable": {"session_id": session_id}},
    )

    # 웹 검색을 통해 보고서 파일 업데이트
    result = agent_with_chat_history.stream(
        {
            "input": "이번에는 삼성전자가 개발한 `생성형 AI` 와 관련된 정보들을 웹 검색하고, 검색한 결과를 정리해 주세요. "
            "한글로 작성해주세요."
            "다음으로는 `report.md` 파일을 열어서 기존의 내용을 읽고, 웹 검색하여 찾은 정보를 이전에 작성한 형식에 맞춰 뒷 부분에 추가해 주세요. \n\n"
            "#작성방법: \n"
            "1. markdown header 2 크기로 적절한 제목을 작성하세요. \n"
            "2. 정보의 출처(url)를 기입하세요(예시: 출처: 네이버 지식백과). \n"
            "3. 정리된 웹검색 내용을 작성하세요. \n"
            "4. 작성이 완료되면 파일을 `report.md` 에 저장하세요. \n"
            "5. 마지막으로 저장한 `report.md` 파일을 읽어서 출력해 주세요. \n"
        },
        config={"configurable": {"session_id": session_id}},
    )

    # 보고서 작성을 요청합니다.
    result = agent_with_chat_history.stream(
        {
            "input": "`report.md` 파일을 열어서 안의 내용을 출력하세요. "
            "출력된 내용을 바탕으로, 전문적인 수준의 보고서를 작성하세요. "
            "보고서는 총 3개의 섹션으로 구성되어야 합니다:\n"
            "1. 개요: 보고서 abstract 를 300자 내외로 작성하세요.\n"
            "2. 핵심내용: 보고서의 핵심 내용을 작성하세요. 정리된 표를 markdown 형식으로 작성하여 추가하세요. "
            "3. 최종결론: 보고서의 최종 결론을 작성하세요. 출처(파일명, url 등)을 표시하세요."
            "마지막으로 작성된 결과물을 `report-2.md` 파일에 저장하세요."
        },
        config={"configurable": {"session_id": session_id}},
    )

    # 이미지 생성을 요청합니다.
    result = agent_with_chat_history.stream(
        {
            "input": "`report-2.md` 파일을 열어서 안의 내용을 출력하세요. "
            "출력된 내용에 어울리는 이미지를 생성하세요. "
            "생성한 이미지의 url 을 markdown 형식으로 보고서의 가장 상단에 추가하세요. "
            "마지막으로 작성된 결과물을 `report-3.md` 파일에 저장하세요."
        },
        config={"configurable": {"session_id": session_id}},
    )