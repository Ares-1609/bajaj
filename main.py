from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# Import the core logic from your other file
from processing_logic import process_document_and_questions

# Initialize the FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="An API that takes a document URL and questions, and returns AI-generated answers.",
    version="1.0.0"
)

# Define the request body model for automatic validation
class RAGRequest(BaseModel):
    documents: HttpUrl  # Pydantic will validate this is a valid URL
    questions: List[str]

# Define the response body model
class RAGResponse(BaseModel):
    answers: List[str]

# Define the API endpoint
@app.post("/hackrx/run", response_model=RAGResponse, tags=["Q&A"])
async def run_rag_pipeline(request: RAGRequest, authorization: Optional[str] = Header(None)):
    """
    This endpoint processes a document from a URL and answers a list of questions based on its content.
    """
    # You can add authorization logic here if needed
    # For now, we'll just check if the header exists
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        # Call the processing function with the validated data
        result = process_document_and_questions(str(request.documents), request.questions)
        return result
    except Exception as e:
        # Return a more detailed error for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

# Add a root endpoint for health checks
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running"}