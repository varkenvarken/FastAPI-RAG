import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# This is a workaround for the sqlite3 module in Python 3.11
# to use pysqlite3 instead of the built-in sqlite3 module.
# This is necessary for compatibility with the ChromaDB library
# when using Python 3.11 and above.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb import HttpClient
from chromadb.utils import embedding_functions
from typing import List, Dict
import httpx
import json
from ollama import Client
from contextlib import asynccontextmanager

@asynccontextmanager
async def startup_event(app: FastAPI):
    client = Client(host=f'http://{OLLAMA_HOST}:{OLLAMA_PORT}')
    print("Checking and pulling Ollama models...")
    try:
        client.show(model='nomic-embed-text')
        print("nomic-embed-text found.")
    except Exception:
        print("nomic-embed-text not found, pulling...")
        client.pull(model='nomic-embed-text')
        print("nomic-embed-text pulled.")

    try:
        client.show(model='llama3')
        print("llama3 found.")
    except Exception:
        print("llama3 not found, pulling...")
        client.pull(model='llama3')
        print("llama3 pulled.")
    yield
    client.close()

app = FastAPI(lifespan=startup_event)

# ChromaDB Settings
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
COLLECTION_NAME = "my_documents"

# Ollama Settings
OLLAMA_HOST = "ollama"
OLLAMA_PORT = 11434
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:latest"  # Assuming llama3:latest is available in your Ollama

# Initialize ChromaDB client
chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Initialize embedding function using Ollama
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
    model_name=EMBEDDING_MODEL,
)

# Get or create the collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ollama_ef
)

class DocumentInput(BaseModel):
    id: str
    content: str

class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    relevant_ids: List[str]



async def get_ollama_response(prompt: str, model: str):
    async with httpx.AsyncClient() as client:
        data = {
            "prompt": prompt,
            "model": model,
            "stream": False,
        }
        print(data)

        response = await client.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate", json=data)
        
        print(response.status_code)
        print("JSON")
            
        # response.raise_for_status()
        return response.json()['response']

@app.post("/document")
async def process_document(doc_input: DocumentInput):
    try:
        collection.add(
            ids=[doc_input.id],
            documents=[doc_input.content],
        )
        return {"message": f"Document with id '{doc_input.id}' processed and stored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_input: QueryInput):
    try:
        results = collection.query(
            query_texts=[query_input.query],
            n_results=3  # You can adjust the number of results
        )
        if results and results['ids'] and results['documents']:
            context = "\n".join(results['documents'][0])
            prompt = f"Based on the following context: '{context}', answer the query: '{query_input.query}'"
            answer = await get_ollama_response(prompt, LLM_MODEL)
            print(f"Answer from LLM: {answer}")
            return QueryResponse(answer=answer, relevant_ids=results['ids'][0])
        else:
            return QueryResponse(answer="No relevant documents found.", relevant_ids=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def pull_models():
    client = Client(host='http://localhost:11434')  # Assuming Ollama is running locally

    print("Pulling nomic-embed-text...")
    try:
        await client.pull(model='nomic-embed-text')
        print("nomic-embed-text pulled successfully.")
    except Exception as e:
        print(f"Error pulling nomic-embed-text: {e}")

    print("\nPulling llama3...")
    try:
        await client.pull(model='llama3')
        print("llama3 pulled successfully.")
    except Exception as e:
        print(f"Error pulling llama3: {e}")
