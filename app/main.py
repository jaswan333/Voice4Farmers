from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.rag_pipeline import rag_pipeline
from app.vector_store import load_vector_store
from typing import Optional

app = FastAPI(title="Agri RAG API")
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "https://voice4farmers-api.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_vector_store()


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.get("/")
def root():
    return {"status": "Agri RAG API Running"}


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = rag_pipeline(request.question, request.session_id)
        return result
    except Exception as e:
        print("PIPELINE ERROR:", str(e))
        return {
            "answer": "Internal server error. Please try again.",
            "confidence": 0.0
        }