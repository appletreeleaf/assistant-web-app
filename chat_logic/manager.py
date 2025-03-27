from typing import List, Optional
from fastapi import UploadFile
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from fastapi import UploadFile
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import (get_vectorstore, get_agent_executor, get_web_loader,
                   generate_questions, remove_sources, process_documents,
                   get_session_history, process_url)

async def handle_schedule(user_input: str,
                          session_id: str,
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
        description="If input query contains a request for information, please use this tool."
    )
    tools = [search]

    # 프롬프트 생성
    # 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "The user will input their schedule. You should remember the schedule and use it in future conversations. "
                "If you need information, you can use the agent tool."
                "Please answer in Korean."
            ),
            ("placeholder", "{chat_history}"),
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
        verbose=False
    )
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
    
    # 특수문자 제거
    answer = remove_sources(response["output"])
    
    # 응답을 대화 내역에 추가
    session_history.add_message({"role": "assistant", "content": answer})
    # AgentExecutor 실행
    answer = response["output"]
    print("Agent 실행 결과:")
    print(type(answer), answer)
    
    return {"response": answer}
