from typing import List, Optional
from fastapi import UploadFile
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import (get_session_history, get_llm)
load_dotenv()

async def handle_translation(user_input: str, user_input2: str, 
                     files: List[UploadFile], session_id: str, 
                     url: Optional[str], usage: str):
    # 입력받은 요소들을 출력
    print(f"User input: {user_input}")
    print(f"User input2: {user_input2}")
    print(f"Session ID: {session_id}")
    print(f"URL: {url}")
    print(f"Selected usage: {usage}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 번역 전문가입니다."
                "입력 텍스트를 {target_language}로 번역해주세요"
                "번역이 자연스럽도록 문장의 흐름과 맥락을 고려해 번역을 진행하세요."

            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    # chain 정의
    llm = get_llm()
    translation_chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        translation_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    response = chain_with_history.invoke(
        {
            "input": user_input, 
            "target_language": user_input2,
        },
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Response: {response}")
    return {"response": response}