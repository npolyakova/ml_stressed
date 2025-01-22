import aiofiles
from fastapi import FastAPI, UploadFile
from starlette.responses import FileResponse

app = FastAPI()

@app.get("/")
def index():
    return FileResponse('../../docs/index.html')

@app.get("/game")
def index():
    return FileResponse('../../docs/zapis.html')

@app.get("/api/words")
async def get_words():
    with open('words.txt', 'r', encoding='utf-8') as file:
        data = file.read()
        return {
            "words": data.split()
        }

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile, word:str):
    async with aiofiles.open('src/app/download.mp3', 'wb') as out_file:
     content = await file.read()  # async read
     await out_file.write(content)  # async write
    return {"Result": word}


