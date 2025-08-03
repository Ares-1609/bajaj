import os
import time
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from urllib.parse import urlparse, unquote




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
    title="Document Q&A API",
    description="Processes a document from a URL and answers questions.",
    version="1.0.0"
)

# --- Pydantic Models for API Data Validation ---
class RAGRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RAGResponse(BaseModel):
    answers: List[str]

#--- Core Processing Logic ---
def setup_vector_store_and_rag_chain(document_url: str):
    """Downloads, chunks, embeds, and indexes a document, then returns the RAG chain."""
    local_pdf_path = "temp_document.pdf"
    try:
        # 1. Download and chunk the document
        print(f"Downloading document from {document_url}")
        doc_response = requests.get(str(document_url))
        doc_response.raise_for_status()
        with open(local_pdf_path, 'wb') as f:
            f.write(doc_response.content)
        
        print(f"📄 Loading document from: {local_pdf_path}")
        loader = PyPDFLoader(local_pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, length_function=len)
        chunks = splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks.")

        # 2. Initialize Pinecone and clear index
        print("\n🚀 Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME in pc.list_indexes().names():
            print(f"🧹 Clearing existing index: {PINECONE_INDEX_NAME}...")
            pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True)
        else:
            print(f"📦 Creating index: {PINECONE_INDEX_NAME}")
            pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine", spec=PodSpec(environment=PINECONE_ENV, pod_type="p1.x1"))
            time.sleep(10)

        # 3. Use the most accurate embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        
        # 4. Create and populate the vector store
        print(f"➕ Embedding and adding {len(chunks)} chunks to Pinecone...")
        vector_store = PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)
        print("✅ Documents embedded and indexed.")

        # --- THE FIX IS HERE ---
        # Add a short delay to allow the Pinecone index to become fully consistent
        print("⏱️ Waiting for index to stabilize...")
        time.sleep(5) 
        
        # 5. Set up the retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"🔍 Retriever initialized.")

        # 6. Set up the LLM and RAG chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)
       prompt = ChatPromptTemplate.from_template("""
You are a clause-extraction AI. Your sole task is to find the single sentence or phrase in the context that directly answers the question.
- Your response must be a single line.
- Do not add any introductory words like "According to the document...".
- Quote the answer directly from the text whenever possible.

Context:
{context}

Question:
{question}

Answer:
""")
        rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
        print("✅ RAG Chain ready.")
        return rag_chain

    finally:
        # Clean up the downloaded file
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)



# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: RAGRequest, authorization: Optional[str] = Header(None)):
    """
    Receives a document URL and questions, processes them, and returns answers.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        # This single call does all the work for the request
        rag_chain = setup_vector_store_and_rag_chain(request.documents)
        
        answers = []
        print("\n" + "=" * 60)
        for question in request.questions:
            print(f"🤔 Processing question: {question}")
            response = rag_chain.invoke({"query": question})
            answers.append(response["result"])
        print("=" * 60 + "\n")
        
        return RAGResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running"}
