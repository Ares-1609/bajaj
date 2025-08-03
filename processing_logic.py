import os
import time
import requests
from dotenv import load_dotenv

# LangChain and Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# --- Load environment variables ---
# For deployment, we'll use environment variables set on the server
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# (All the helper functions: load_and_chunk_data, get_retriever, etc., go here)
# For brevity, they are included directly in the main processing function below.

def process_document_and_questions(document_url: str, questions: list):
    """
    Takes a document URL and a list of questions, performs RAG, and returns the answers.
    """
    local_pdf_path = "downloaded_policy.pdf"
    try:
        print(f"Downloading document from {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()

        with open(local_pdf_path, 'wb') as f:
            f.write(response.content)

        # --- Internal Helper Functions ---
        def load_and_chunk_data(file_path: str):
            print(f"📄 Loading document from: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, length_function=len, add_start_index=True)
            chunks = splitter.split_documents(documents)
            print(f"✅ Split into {len(chunks)} chunks.")
            return chunks

        def create_vector_store_pinecone(chunks):
            print("\n🚀 Initializing Pinecone...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            if PINECONE_INDEX_NAME in pc.list_indexes().names():
                print(f"🧹 Clearing existing index: {PINECONE_INDEX_NAME}...")
                index_to_clear = pc.Index(PINECONE_INDEX_NAME)
                index_to_clear.delete(delete_all=True)
            else:
                print(f"📦 Creating index: {PINECONE_INDEX_NAME}")
                pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine", spec=PodSpec(environment=PINECONE_ENV, pod_type="p1.x1"))
                print("⏳ Waiting for index to be ready...")
                time.sleep(10)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            print("🔗 Getting a handle to the Pinecone index...")
            index = pc.Index(PINECONE_INDEX_NAME)
            vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key='text')
            print(f"➕ Adding {len(chunks)} chunks to the index...")
            vector_store.add_documents(documents=chunks)
            print("✅ Documents embedded and added to Pinecone index.")
            return vector_store

        def get_retriever(vector_store, search_kwargs={"k": 5}):
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            print(f"🔍 Retriever initialized (top {search_kwargs['k']} results).")
            return retriever

        def setup_rag_chain_gemini(retriever, llm_model_name="gemini-1.5-flash-latest"):
            print(f"\n🤖 Setting up Gemini RAG Chain with model: {llm_model_name}")
            llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.3, google_api_key=GOOGLE_API_KEY)
            prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the following context to answer the user's question. Give the answer anyhow from the provided document.
Context: {context}
Question: {question}
Answer:""")
            rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": prompt})
            print("✅ RAG Chain ready.")
            return rag_chain

        # --- Main Logic Execution ---
        chunks = load_and_chunk_data(local_pdf_path)
        vector_store = create_vector_store_pinecone(chunks)
        retriever = get_retriever(vector_store)
        rag_chain = setup_rag_chain_gemini(retriever)

        answers = []
        print("\n" + "=" * 60)
        for i, question in enumerate(questions):
            print(f"🤔 Processing question {i+1}/{len(questions)}: {question}")
            response = rag_chain.invoke({"query": question})
            answers.append(response["result"])
        print("=" * 60 + "\n")
        
        return {"answers": answers}

    finally:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
