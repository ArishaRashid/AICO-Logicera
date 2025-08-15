"""
FastAPI application for AICO Web Summarization Agent
Provides API endpoints for summarizing web content using advanced chain summarization techniques
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Import AICO agent components
from aico_agent.agent_orchestrator import (
    build_web_summarization_agent, 
    summarize_web_content,
    get_summarization_techniques,
    chat_with_agent_simple
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AICO Web Summarization API",
    description="Advanced web content summarization using chain summarization techniques with OpenAI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SummarizeRequest(BaseModel):
    """Request model for web summarization."""
    url: HttpUrl
    technique: Optional[str] = "adaptive"
    
    @validator('technique')
    def validate_technique(cls, v):
        """Validate that the technique is supported."""
        if v is not None:
            available_techniques = get_summarization_techniques().keys()
            if v not in available_techniques:
                raise ValueError(f"Unsupported technique. Available: {', '.join(available_techniques)}")
        return v

class SummarizeResponse(BaseModel):
    """Response model for web summarization."""
    summary: str
    main_topic: str
    technique_used: str
    content_length: int
    success: bool = True
    message: Optional[str] = None

class ChatRequest(BaseModel):
    """Request model for chat with the agent."""
    message: str
    url: Optional[HttpUrl] = None
    
    @validator('message')
    def validate_message(cls, v):
        """Validate that the message is not empty."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    version: str

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        message="AICO Web Summarization API is running with OpenAI models",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Service is operational",
        version="1.0.0"
    )

@app.get("/techniques", response_model=Dict[str, str])
async def get_techniques():
    """Get available summarization techniques and their descriptions."""
    try:
        techniques = get_summarization_techniques()
        logger.info(f"✅ Retrieved {len(techniques)} summarization techniques")
        return techniques
    except Exception as e:
        logger.error(f"❌ Error retrieving techniques: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve summarization techniques: {str(e)}"
        )

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_webpage(request: SummarizeRequest):
    """
    Summarize a webpage using advanced chain summarization techniques with OpenAI models.
    
    Args:
        request: SummarizeRequest containing URL and optional technique
        
    Returns:
        SummarizeResponse with summary, main topic, and metadata
    """
    try:
        logger.info(f"🔍 Starting summarization for URL: {request.url}")
        logger.info(f"�� Using technique: {request.technique}")
        
        # Validate environment variables
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        
        # Perform the summarization
        result = summarize_web_content(
            url=str(request.url),
            technique=request.technique
        )
        
        logger.info("✅ Webpage summarization completed successfully")
        
        return SummarizeResponse(
            summary=result["summary"], 
            main_topic=result["main_topic"],
            technique_used=result["technique_used"],
            content_length=result["content_length"],
            success=True,
            message="Webpage summarized successfully using OpenAI models"
        )
        
    except ValueError as e:
        logger.warning(f"⚠️ Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error during summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )

@app.post("/summarize/async", response_model=Dict[str, str])
async def summarize_webpage_async(
    request: SummarizeRequest, 
    background_tasks: BackgroundTasks
):
    """
    Asynchronously summarize a webpage (for long-running tasks).
    
    Args:
        request: SummarizeRequest containing URL and optional technique
        background_tasks: FastAPI background tasks
        
    Returns:
        Immediate response with task ID
    """
    try:
        logger.info(f"🔍 Starting async summarization for URL: {request.url}")
        
        # For now, return immediate response
        # In a production system, you'd implement proper async task handling
        # with Redis, Celery, or similar
        
        return {
            "message": "Async summarization started",
            "task_id": f"summarize_{hash(str(request.url))}",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting async summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async summarization: {str(e)}"
        )

@app.post("/chat", response_model=Dict[str, Any])
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the AI agent about previously summarized content.
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        Agent's response with conversation context
    """
    try:
        logger.info(f"💬 Chat request: {request.message[:100]}...")
        
        # Validate environment variables
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        
        # Build the agent with conversation memory
        llm, memory = build_web_summarization_agent()
        
        # Get response using simple chat function
        url_str = str(request.url) if request.url else None
        response = chat_with_agent_simple(request.message, url_str, llm, memory)
        
        logger.info("✅ Chat response generated successfully")
        
        return {
            "response": response["response"],
            "chat_history": response.get("chat_history", []),
            "success": True,
            "message": "Chat response generated successfully"
        }
        
    except ValueError as e:
        logger.warning(f"⚠️ Validation error in chat: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error during chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return ErrorResponse(
        success=False,
        error=exc.detail,
        message="Request failed"
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    return ErrorResponse(
        success=False,
        error="Internal server error",
        message="An unexpected error occurred"
    )

# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("🚀 Starting AICO Web Summarization API with OpenAI models...")
    
    # Check required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        logger.warning("⚠️ API may not function properly without these variables")
    else:
        logger.info("✅ All required environment variables are set")
    
    logger.info("✅ AICO Web Summarization API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🛑 Shutting down AICO Web Summarization API...")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
