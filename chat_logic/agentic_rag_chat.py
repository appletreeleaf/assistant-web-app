from typing import List, Optional
from fastapi import UploadFile
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import (get_vectorstore, get_agent_executor, 
                   generate_questions, remove_sources, process_documents,
                   get_session_history)


async def handle_rag(user_input: str, files: List[UploadFile], 
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

async def reset_session(session_id: str):
    session_history = get_session_history(session_id)
    session_history.clear()  # 세션 내역 삭제 (예시)
    return {"message": "Session reset successful"}
