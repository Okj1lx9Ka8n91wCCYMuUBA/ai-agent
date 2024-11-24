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

llm = Ollama(model="gemma2:9b", base_url=base_url)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

recommendation_api_url = "http://127.0.0.1:8809/recommendations"


def convert_to_documents(doc_list):
    documents = []
    for doc in doc_list:
        content = doc.get('content', '')
        title = doc.get('title', '')
        full_text = f"{title}\n{content}"
        documents.append(Document(text=full_text))
    return documents

def initialize_index():
    documents = []
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
    startup_details: dict

class QueryResponse(BaseModel):
    answer: str
    recommendations: list

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Endpoint to query documents using RAG and fetch recommendations."""
    try:
        # Query LlamaIndex
        query_engine = index.as_query_engine()
        response = query_engine.query(query_request.question)

        # Fetch recommendations
        recommendation_payload = query_request.startup_details
        try:
            recommendation_response = requests.post(
                recommendation_api_url,
                json=recommendation_payload,
                headers={"Content-Type": "application/json"}
            )
            recommendation_response.raise_for_status()
            recommendations = recommendation_response.json().get("recommendations", [])
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")

        return QueryResponse(answer=response.response, recommendations=recommendations)
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