from typing import List, Optional
from fastapi import UploadFile
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import (
    get_vectorstore, 
    get_agent_executor,
    generate_questions, 
    remove_sources, 
    process_documents,
    get_session_history, 
    process_url
)


async def handle_document(
                    user_input: str, files: List[UploadFile], 
                    session_id: str, usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"Selected usage: {usage}")
    
    doc_list = []

    # file, url을 문서로 변환
    if files:
        doc_list = await process_documents(doc_list=doc_list, files=files, session_id=session_id)

    # 최종 문서 일부 내용
    print(f"2nd Content: {doc_list[1].page_content[:100]}")
    print(f"3rd Content: {doc_list[2].page_content[:100]}")
    if doc_list:
        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        
        retriever = vector_store.as_retriever()
    else:
        retriever = None

    agent_executor = get_agent_executor(usage, retriever)

    # Create agent with chat history
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
    
    # 관련질문 생성
    references = generate_questions(answer)
    
    return {"response": answer, "references": references}


async def handle_url(user_input: str, session_id: str, 
                     url: Optional[str], usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"URL: {url}")
    print(f"Selected usage: {usage}")
    
    doc_list = []

    # url을 문서로 변환
    if url:
        doc_list = await process_url(doc_list=doc_list, url=url, session_id=session_id)

    # 최종 문서 일부 내용
    print(f"2nd Content: {doc_list[1].page_content[:100]}")
    print(f"3rd Content: {doc_list[2].page_content[:100]}")
    if doc_list:
        vector_store = get_vectorstore(doc_list)
        if vector_store is None:
            raise ValueError("Error: vector_store is not initialized.")
        
        retriever = vector_store.as_retriever()
    else:
        retriever = None

    agent_executor = get_agent_executor(usage, retriever)

    # Create agent with chat history
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
    
    # 관련질문 생성
    references = generate_questions(answer)
    
    return {"response": answer, "references": references}
