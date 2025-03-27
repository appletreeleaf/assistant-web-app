import os
from typing import List, Optional
from fastapi import UploadFile
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from utils import (generate_questions, get_web_loader, validate_url,
                   get_session_history, get_llm, get_prompt, get_document_loaders)

from langchain_core.runnables import chain


@chain
def map_reduce_chain(docs, session_id):
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
    reduce_chain_with_history = RunnableWithMessageHistory(
        reduce_chain,
        get_session_history,  # 직접 get_session_history 함수 전달
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    return reduce_chain_with_history.invoke(
        {"doc_summaries": doc_summaries, "language": "Korean"},
        config={"configurable": {"session_id": session_id}})

UPLOAD_DIRECTORY = "uploads"
uploaded_files = {} 

async def handle_summarize(user_input: str, files: List[UploadFile], 
                      session_id: str, url: Optional[str], 
                      usage: str):
    # url 유효성 검사
    print(f"handle_summarize's url: {url}")

    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"URL: {url}")
    print(f"Selected usage: {usage}")
    
    llm = get_llm()
    doc_list = []
    if files:
        for file in files:
            file_name = os.path.join(UPLOAD_DIRECTORY, file.filename)  # 업로드할 디렉토리 지정 
            with open(file_name, "wb") as f:
                f.write(await file.read())
            print(f"Uploaded file: {file.filename}")

            # 파일의 확장자에 따라 적절한 로더 사용
            loader = get_document_loaders(file_name)
            docs = loader.load()
            doc_list.extend(docs)
            if session_id not in uploaded_files:
                uploaded_files[session_id] = []
            uploaded_files[session_id].append(file_name)  # 세션 ID에 파일 경로 추가
            os.remove(file_name)
    if url:
        loader = get_web_loader(url)
        docs = loader.load()
        doc_list.extend(docs)

        if session_id not in uploaded_files:
            uploaded_files[session_id] = []
        uploaded_files[session_id].append(url)  # 세션 ID에 파일 경로 추가
        print(f"Loaded documents from URL: {url}")
    
    if doc_list:
        doc_len = "long" if len(doc_list) > 10 else "short"

        if doc_len == "long":
            response = map_reduce_chain.invoke(doc_list, session_id)
        else:
            stuff_chain = create_stuff_documents_chain(llm=llm, prompt=get_prompt()["summarize"][doc_len])
            reduce_chain_with_history = RunnableWithMessageHistory(
                stuff_chain,
                get_session_history,  # 직접 get_session_history 함수 전달
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            response = reduce_chain_with_history.invoke(
                {"context": doc_list},
                config={"configurable": {"session_id": session_id}})
    else:
        prompt = get_prompt()["general_prompt"]
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,  # 직접 get_session_history 함수 전달
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        response = chain_with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
            )

    # session id 생성
    session_history = get_session_history(session_id)
    # 대화 내역 추가
    session_history.add_message({"role": "user", "content": user_input})
    session_history.add_message({"role": "assistant", "content": response})
        
    # 관련질문 생성
    references = generate_questions(response)
    print(response)
    return {"response": response, "references": references}