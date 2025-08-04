import os
import time
import requests
import uvicorn
import asyncio
import hashlib
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from dotenv import load_dotenv

# LangChain and Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# --- Load Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Optimized Document Q&A API",
    description="Uses a single endpoint with a caching mechanism to provide fast, accurate answers.",
    version="3.0.0"
)

# --- Pydantic Models for API Data Validation ---
class RAGRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RAGResponse(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: RAGRequest, authorization: Optional[str] = Header(None)):
    """
    Receives a document URL and questions. It checks if the document is cached;
    if not, it indexes it. Then, it answers the questions in parallel.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    document_url = str(request.documents)
    # Create a unique, deterministic ID for the document URL to use as a namespace
    namespace_id = hashlib.sha256(document_url.encode()).hexdigest()

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create the index if it doesn't already exist
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Index not found. Creating index: {PINECONE_INDEX_NAME}")
            pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine", spec=PodSpec(environment=PINECONE_ENV))
            time.sleep(10) # Wait for index to be ready
        
        index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

        # Check if the document is already cached in the namespace
        query_res = index.query(vector=[0]*768, top_k=1, namespace=namespace_id)
        
        if not query_res['matches']:
            print(f"🚀 Document not found in cache. Starting one-time indexing for namespace: {namespace_id[:10]}...")
            local_pdf_path = "temp_document.pdf"
            try:
                # Download, chunk, and index the document into the specific namespace
                doc_response = requests.get(document_url)
                doc_response.raise_for_status()
                with open(local_pdf_path, 'wb') as f: f.write(doc_response.content)

                loader = PyPDFLoader(local_pdf_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, length_function=len)
                chunks = splitter.split_documents(documents)
                
                print(f"➕ Embedding and adding {len(chunks)} chunks to Pinecone namespace...")
                PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME, namespace=namespace_id)
                print("✅ Indexing complete.")
            finally:
                if os.path.exists(local_pdf_path): os.remove(local_pdf_path)
        else:
            print(f"✅ Document found in cache for namespace: {namespace_id[:10]}. Skipping indexing.")

        # Now, perform the fast querying using the correct namespace
        vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace=namespace_id)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        
        # --- PROMPT HAS BEEN UPDATED AS PER YOUR REQUEST ---
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the user's question. 
Give the answer anyhow from the provided document. If the answer isn't in the context, say you cannot find the answer in the document.

Context:
{context}

Question:
{question}

Answer:
""")
        
        rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})

        # Process questions in parallel for speed
        async def get_answer(question: str):
            print(f"🤔 Processing question: {question}")
            response = await rag_chain.ainvoke({"query": question})
            return response["result"]

        tasks = [get_answer(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        return RAGResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running"}
