import os
import time
import requests
import uvicorn
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
# For deployment, set these in your hosting environment (e.g., Render)
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

# --- Core Processing Logic (Kept as is for accuracy) ---
def load_and_chunk_data(file_path: str):
    print(f"📄 Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, length_function=len)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

def setup_vector_store_and_rag_chain(chunks):
    print("\n🚀 Initializing services and building RAG chain...")
    
    # 1. Initialize Pinecone and clear index for a fresh start
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        print(f"🧹 Clearing existing index: {PINECONE_INDEX_NAME}...")
        index_to_clear = pc.Index(PINECONE_INDEX_NAME)
        index_to_clear.delete(delete_all=True)
    else:
        print(f"📦 Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine", spec=PodSpec(environment=PINECONE_ENV, pod_type="p1.x1"))
        time.sleep(10)

    # 2. Use the most accurate embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    # 3. Create and populate the vector store
    print(f"➕ Embedding and adding {len(chunks)} chunks to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)
    print("✅ Documents embedded and indexed.")

    # 4. Set up the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print(f"🔍 Retriever initialized.")

    # 5. Set up the LLM and RAG chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the following context to answer the user's question. Give the answer anyhow from the provided document.
Context: {context}
Question: {question}
Answer:""")
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
    print("✅ RAG Chain ready.")
    return rag_chain

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: RAGRequest, authorization: Optional[str] = Header(None)):
    """
    Receives a document URL and questions, processes them, and returns answers.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    local_pdf_path = "temp_document.pdf"
    try:
        # Download the document
        print(f"Downloading document from {request.documents}")
        doc_response = requests.get(str(request.documents))
        doc_response.raise_for_status()
        with open(local_pdf_path, 'wb') as f:
            f.write(doc_response.content)

        # Process the document and set up the RAG chain
        chunks = load_and_chunk_data(local_pdf_path)
        rag_chain = setup_vector_store_and_rag_chain(chunks)
        
        # Process all questions
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
    finally:
        # Clean up the downloaded file
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running"}

# --- To run this locally ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
