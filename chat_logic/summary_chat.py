import os
from typing import List, Optional
from fastapi import UploadFile
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from utils import (generate_questions, remove_sources, process_documents,
                   get_session_history, get_llm, get_prompt, get_document_loaders)

from langchain_core.runnables import chain


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

UPLOAD_DIRECTORY = "uploads"
uploaded_files = {} 

async def handle_summarize(user_input: str, files: List[UploadFile], 
                      session_id: str, url: Optional[str], 
                      usage: str):
    # 입력받은 요소들을 로그에 남기기
    print(f"User input: {user_input}")
    print(f"Session ID: {session_id}")
    print(f"URL: {url}")
    print(f"Selected usage: {usage}")
    
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
    
    if doc_list:
        doc_len = "long" if len(doc_list) > 10 else "short"
        prompt = get_prompt(usage)[doc_len]

        if doc_len == "long":
            response = map_reduce_chain.invoke(doc_list)
        else:
            chain = create_stuff_documents_chain(llm=get_llm(), prompt=get_prompt(usage)[doc_len])
            response = chain.invoke({"context": doc_list})

        # 대화 내역에 사용자 입력 추가
        session_history = get_session_history(session_id)
        session_history.add_message({"role": "user", "content": user_input})
        # 응답을 대화 내역에 추가
        session_history.add_message({"role": "assistant", "content": response})
        
        # 관련질문 생성
        references = generate_questions(response)
        print(response)
    return {"response": response, "references": references}