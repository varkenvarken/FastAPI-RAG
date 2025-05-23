import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# This is a workaround for the sqlite3 module in Python 3.11
# to use pysqlite3 instead of the built-in sqlite3 module.
# This is necessary for compatibility with the ChromaDB library
# when using Python 3.11 and above.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb import HttpClient, Collection
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

class DocumentInput(BaseModel):
    id: str
    content: str
    collection: str

class QueryInput(BaseModel):
    query: str
    collection: str

class QueryResponse(BaseModel):
    answer: str
    relevant_ids: List[str]
    collection: str

def get_chroma_collections_info(chroma_client: HttpClient) -> list[dict]:
    """
    Retrieves a list of collections from ChromaDB with their names and document counts.

    Args:
        chroma_client: An initialized ChromaDB HttpClient instance.

    Returns:
        A list of dictionaries, where each dictionary contains 'name' (str)
        and 'document_count' (int) for a collection.
        Returns an empty list if no collections are found or an error occurs.
    """
    collections_info = []
    try:
        # List all collections available in the ChromaDB instance
        collections: list[Collection] = chroma_client.list_collections()

        if not collections:
            print("No collections found in ChromaDB.")
            return []

        # Iterate through each collection to get its name and document count
        for collection in collections:
            collection_name = collection.name
            document_count = collection.count() # Get the number of documents in the collection
            collections_info.append({
                "name": collection_name,
                "document_count": document_count
            })
    except Exception as e:
        print(f"An error occurred while fetching ChromaDB collection info: {e}")
        # Depending on desired error handling, you might re-raise the exception
        # or return a specific error indicator. For now, we'll return an empty list.
        return []
    return collections_info


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
        collection = chroma_client.get_or_create_collection(name=doc_input.collection, embedding_function=ollama_ef)
        collection.add(
            ids=[doc_input.id],
            documents=[doc_input.content],
        )
        return {"message": f"Document with id '{doc_input.id}' processed and stored in {doc_input.collection}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_input: QueryInput):
    try:
        collection = chroma_client.get_or_create_collection(name=query_input.collection, embedding_function=ollama_ef)
        results = collection.query(
            query_texts=[query_input.query],
            n_results=3  # You can adjust the number of results
        )
        if results and results['ids'] and results['documents']:
            context = "\n".join(results['documents'][0])
            prompt = f"Based on the following context: '{context}', answer the query: '{query_input.query}'"
            answer = await get_ollama_response(prompt, LLM_MODEL)
            print(f"Answer from LLM: {answer}")
            return QueryResponse(answer=answer, relevant_ids=results['ids'][0], collection=query_input.collection)
        else:
            return QueryResponse(answer="No relevant documents found.", relevant_ids=[], collection=query_input.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    try:
        collections_info = get_chroma_collections_info(chroma_client)
        return {"status": "OK", "collections": collections_info}
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
