import os
import time
import requests
import uvicorn
import asyncio
import hashlib
import json
import re
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from dotenv import load_dotenv

# LangChain and Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# --- Load Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- FastAPI Initialization ---
app = FastAPI(
    title="Intelligent Document Q&A API",
    description="Handles structured and unstructured queries on policy documents using LLM + RAG.",
    version="3.2.0"
)

# --- Pydantic Models ---
class RAGRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    decision: str
    amount: Optional[str] = None
    justification: str
    parsed_query: Optional[dict] = None

class RAGResponse(BaseModel):
    answers: List[Answer]

# --- Utility: Determine if query is structured ---
def is_structured_query(query: str) -> bool:
    pattern = r'\b\d{1,3}[ ]?(years?|yrs?|yo|year[- ]?old)?\b|male|female|surgery|policy|months?|procedure|location|in\s+\w+'
    return re.search(pattern, query.lower()) is not None

# --- Utility: Parse free-form query using LLM ---
async def parse_input_query(query: str, llm) -> dict:
    prompt = PromptTemplate.from_template("""
You are a smart insurance query parser. Extract structured information from the text.

Return JSON with these fields:
- age (integer or null)
- gender (male/female/other/unknown)
- procedure (string or null)
- location (string or null)
- policy_duration (string or null)

If a field is missing, return null.

Query: "{query}"

Respond ONLY in JSON.
""")
    chain = prompt | llm
    result = await chain.ainvoke({"query": query})
    try:
        return json.loads(result)
    except:
        return {
            "error": "Unable to parse query",
            "raw_response": result
        }

# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: RAGRequest, authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    document_url = str(request.documents)
    namespace_id = hashlib.sha256(document_url.encode()).hexdigest()

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Index not found. Creating index: {PINECONE_INDEX_NAME}")
            pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine", spec=PodSpec(environment=PINECONE_ENV))
            time.sleep(10)

        index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

        query_res = index.query(vector=[0]*768, top_k=1, namespace=namespace_id)

        if not query_res['matches']:
            print(f"📂 No cached index found. Downloading and processing document.")
            local_file_path = "temp_document"
            try:
                doc_response = requests.get(document_url)
                doc_response.raise_for_status()
                with open(local_file_path, 'wb') as f:
                    f.write(doc_response.content)

                if document_url.lower().endswith('.pdf'):
                    loader = PyPDFLoader(local_file_path)
                elif document_url.lower().endswith('.docx'):
                    loader = Docx2txtLoader(local_file_path)
                else:
                    raise HTTPException(status_code=415, detail="Unsupported file type. Provide .pdf or .docx URL.")

                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, length_function=len)
                chunks = splitter.split_documents(documents)

                print(f"📌 Indexing {len(chunks)} document chunks...")
                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME,
                    namespace=namespace_id
                )
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
        else:
            print(f"✅ Document found in cache. Skipping re-indexing.")

        # Retrieval Setup
        vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace=namespace_id)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context from the insurance document to answer the question.

Context:
{context}

Question:
{question}

Answer:""")

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        # Process Questions
        answers = []
        for user_question in request.questions:
            if is_structured_query(user_question):
                print(f"🧠 Structured query detected.")
                structured_info = await parse_input_query(user_question, llm)
            else:
                structured_info = None
                print(f"📄 Fact-based query detected.")

            response = await rag_chain.ainvoke({"query": user_question})
            rag_answer = response["result"]

            decision = "approved" if "covered" in rag_answer.lower() else "refer_to_document"

            answers.append({
                "decision": decision,
                "amount": None,
                "justification": rag_answer,
                "parsed_query": structured_info
            })

        return RAGResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Health Check ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running"}
