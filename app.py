import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from handlers import router  # handler 모듈에서 라우터 가져오기

app = FastAPI()

# static 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# router를 fastapi 애플리케이션에 등록
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
