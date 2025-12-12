from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import uvicorn
import json
from bedrock_service import bedrock_service

app = FastAPI(title="Cloudscape API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ItemCreate(BaseModel):
    name: str
    status: str = "active"

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None

class Item(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime

class BedrockExtractionRequest(BaseModel):
    prompt_template: str
    model_id: Optional[str] = "anthropic.claude-3-sonnet-20240229-v1:0"
    hyperparameters: Optional[Dict[str, Any]] = None

class BedrockExtractionResponse(BaseModel):
    success: bool
    extracted_content: str
    model_used: str
    usage_metrics: Dict[str, int]
    hyperparameters_used: Dict[str, Any]
    document_info: Dict[str, Any]
    processing_time_seconds: Optional[float] = None

# In-memory storage (replace with database in production)
items_db = {}

@app.get("/")
async def root():
    return {"message": "Cloudscape API is running"}

@app.get("/api/items", response_model=List[Item])
async def get_items():
    return list(items_db.values())

@app.post("/api/items", response_model=Item)
async def create_item(item: ItemCreate):
    item_id = str(uuid.uuid4())
    new_item = Item(
        id=item_id,
        name=item.name,
        status=item.status,
        created_at=datetime.now()
    )
    items_db[item_id] = new_item
    return new_item

@app.get("/api/items/{item_id}", response_model=Item)
async def get_item(item_id: str):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.put("/api/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item_update: ItemUpdate):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    existing_item = items_db[item_id]
    update_data = item_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(existing_item, field, value)
    
    return existing_item

@app.delete("/api/items/{item_id}")
async def delete_item(item_id: str):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    del items_db[item_id]
    return {"message": "Item deleted successfully"}

# Bedrock document processing endpoints
@app.post("/api/bedrock/extract", response_model=BedrockExtractionResponse)
async def extract_from_document(
    pdf_file: UploadFile = File(...),
    prompt_template: str = Form(...),
    model_id: str = Form("anthropic.claude-3-sonnet-20240229-v1:0"),
    hyperparameters: str = Form("{}")
):
    """
    Extract information from PDF using AWS Bedrock
    """
    # Validate file type
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(hyperparameters) if hyperparameters else {}
        
        # Read PDF content
        pdf_content = await pdf_file.read()
        
        # Record start time
        start_time = datetime.now()
        
        # Process with Bedrock
        result = await bedrock_service.process_document_with_bedrock(
            pdf_content=pdf_content,
            prompt_template=prompt_template,
            model_id=model_id,
            hyperparameters=hyperparams,
            filename=pdf_file.filename or "document.pdf"
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time_seconds"] = processing_time
        
        return BedrockExtractionResponse(**result)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid hyperparameters JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/bedrock/models")
async def get_available_models():
    """
    Get list of available Bedrock models and their capabilities
    """
    return bedrock_service.get_available_models()

@app.post("/api/bedrock/test-connection")
async def test_bedrock_connection():
    """
    Test Bedrock connection and IAM permissions
    """
    try:
        # Try to list available models to test connection
        models = bedrock_service.get_available_models()
        return {
            "success": True,
            "message": "Bedrock connection successful",
            "available_models": len(models["text_models"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bedrock connection failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=120,  # Keep connections alive longer
        timeout_graceful_shutdown=30
    )