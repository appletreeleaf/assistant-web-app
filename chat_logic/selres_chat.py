from typing import List, Optional
from fastapi import UploadFile
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from utils import (get_vectorstore, get_agent_executor, 
                   generate_questions, remove_sources, process_documents,
                   get_session_history)
from fastapi.responses import JSONResponse

async def handle_selres(user_input: str, files: List[UploadFile], 
                         session_id: str, url: Optional[str], 
                         usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"Selected usage: {usage}")

    load_dotenv()

    search = TavilySearchResults(
        max_results=5,
        include_answer=True,
        include_raw_content=True,
        description="use this tool to search information"
    )
    tools = [search]
    from langchain_core.prompts import ChatPromptTemplate

    # 프롬프트 생성
    # 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "Make sure to use the `search` tool for searching keyword related place."
                "Please include url and Answer in korean."
            ),
            # ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 에이전트 생성
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_openai_tools_agent

    # LLM 정의
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Agent 생성
    agent = create_openai_tools_agent(llm, tools, prompt)

    from langchain.agents import AgentExecutor

    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # AgentExecutor 실행
    result = agent_executor.invoke({"input": user_input})
    answer = result["output"]
    print("Agent 실행 결과:")
    print(answer)


    return {"response": answer}
