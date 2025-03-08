from typing import List, Optional
from fastapi import UploadFile
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from utils import (generate_questions, remove_sources, process_documents,
                   get_session_history, get_llm, get_prompt)

from langchain_core.runnables import chain

@chain
def stuff_chain(docs, prompt):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_chain

@chain
def map_reduce_chain(docs):
    map_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )

    # map prompt 다운로드
    map_prompt = hub.pull("teddynote/map-prompt")

    # map chain 생성
    map_chain = map_prompt | map_llm | StrOutputParser()

    # 첫 번째 프롬프트, ChatOpenAI, 문자열 출력 파서를 연결하여 체인을 생성합니다.
    doc_summaries = map_chain.batch(docs)

    # reduce prompt 다운로드
    reduce_prompt = hub.pull("teddynote/reduce-prompt")
    reduce_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        streaming=True,
    )

    reduce_chain = reduce_prompt | reduce_llm | StrOutputParser()

    return reduce_chain.invoke({"doc_summaries": doc_summaries, "language": "Korean"})


async def handle_summarize(user_input: str, files: List[UploadFile], 
                      session_id: str, url: Optional[str], 
                      usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"URL: {url}")
    print(f"Selected usage: {usage}")
    
    doc_list = []
    # 파일 처리
    if files:
        doc_list = await process_documents(doc_list, files=files, session_id=session_id)
    
    if doc_list:
        doc_len = "long" if len(doc_list) > 10 else "short"
        prompt = get_prompt(usage)[doc_len]

        if doc_len == "long":
            response = map_reduce_chain.invoke(doc_list)
        else:
            chain = stuff_chain(doc_list, prompt)

            # Create agent with chat history
            chain_with_chat_history = RunnableWithMessageHistory(
                chain,
                get_session_history,  # 직접 get_session_history 함수 전달
                input_messages_key="input",
                history_messages_key="chat_history"
            )

        # 에이전트를 통해 응답 생성
            response = chain_with_chat_history.invoke(
                {"context": doc_list},
                config={"configurable": {"session_id": session_id}}  # 세션 ID를 포함하여 호출
            )
        
        # 특수문자 제거
        answer = remove_sources(response)
        

        # 대화 내역에 사용자 입력 추가
        session_history = get_session_history(session_id)
        session_history.add_message({"role": "user", "content": user_input})
        # 응답을 대화 내역에 추가
        session_history.add_message({"role": "assistant", "content": answer})
        
        # 관련질문 생성
        references = generate_questions(answer)
    
    return {"response": answer, "references": references}