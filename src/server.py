from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index import LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index import Document
import requests


server_ip = "81.94.150.39"
server_port = 7869
base_url = f"http://{server_ip}:{server_port}"
# base_url = "ollama:11434" # IF USING DOCKER SETUP


llm = Ollama(model="gemma2:9b", base_url=base_url)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


def fetch_documents():
    api_base_url = "http://andpoint"
    endpoint = "/documents"
    url = api_base_url + endpoint
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        documents = response.json()
        return documents
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching documents: {e}")

# Convert fetched documents to LlamaIndex format
def convert_to_documents(doc_list):
    documents = []
    for doc in doc_list:
        content = doc.get('content', '')
        title = doc.get('title', '')
        full_text = f"{title}\n{content}"
        documents.append(Document(text=full_text))
    return documents

# Initialize index
def initialize_index():
    doc_list = fetch_documents()
    documents = convert_to_documents(doc_list)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

index = initialize_index()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Endpoint to query documents using RAG."""
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query_request.question)
        return QueryResponse(answer=response.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors globally."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected error: {str(exc)}"}
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG REST API"}