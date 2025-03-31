from fastapi import FastAPI

from mediqa.rag.manager import VectorDBManager
from mediqa.rag.reader import Reader
from mediqa.config.core import config

app = FastAPI()
vdb_manager = VectorDBManager(config.rag_config)
reader = Reader(config.reader_config)

@app.get("/generate")
async def generate_response(question: str):
    retrieved_docs = vdb_manager.retrieve(question)
    return reader.generate([question], [retrieved_docs])