from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from handlers import router  # 핸들러에서 라우터 가져오기

app = FastAPI()
# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)  # 라우터 포함

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
