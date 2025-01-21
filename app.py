from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (실제 배포 시에는 특정 출처만 허용하는 것이 좋습니다)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain ChatGPT 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Pydantic 모델 정의
class UserInput(BaseModel):
    user_input: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(user_input: str = "", file: UploadFile = File(None)):
    # 파일이 업로드된 경우 처리
    if file:
        # 파일 내용을 읽기 (예: 텍스트 파일)
        contents = await file.read()
        # 예시로 파일 내용을 문자열로 변환 (필요에 따라 추가 처리 가능)
        file_content = contents.decode('utf-8')
        user_input += f"\nFile content: {file_content}"

    # LangChain을 통해 챗봇의 응답 생성
    response = llm.invoke({"input": user_input})
    return {"response": response["output"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
