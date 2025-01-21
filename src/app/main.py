from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/api/words")
async def get_words():
    with open('words.txt', 'r', encoding='utf-8') as file:
        data = file.read()
        return {
            "words": data.split()
        }

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
