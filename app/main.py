"""
FastAPI application for AICO Web Summarization Agent
Provides API endpoints for summarizing web content using advanced chain summarization techniques
"""

import os
import logging
import time
import uuid
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator
import uvicorn

# Import AICO agent components
from aico_agent.agent_orchestrator import (
    build_web_summarization_agent, 
    summarize_web_content,
    get_summarization_techniques,
    chat_with_agent_simple
)

# Global chat session storage (single user, persistent memory)
# Single global session for maintaining chat history across all requests
global_chat_session = None

def get_or_create_global_chat_session():
    """Get or create a single global chat session with persistent memory."""
    global global_chat_session
    
    if global_chat_session is None:
        logger.info("🆕 Creating new global chat session")
        llm, memory = build_web_summarization_agent()
        global_chat_session = {
            "llm": llm,
            "memory": memory,
            "created_at": time.time(),
            "last_activity": time.time()
        }
        logger.info("✅ Global chat session created successfully")
    else:
        # Update last activity
        global_chat_session["last_activity"] = time.time()
        logger.info("📱 Using existing global chat session")
    
    return global_chat_session["llm"], global_chat_session["memory"]

def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old global chat session if needed."""
    global global_chat_session
    
    if global_chat_session is not None:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        if current_time - global_chat_session["last_activity"] > max_age_seconds:
            logger.info("🧹 Cleaning up old global chat session")
            global_chat_session = None
        else:
            logger.info("📱 Global chat session is still active")
    else:
        logger.info("📱 No global chat session to clean up")

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
    
    @field_validator('technique')
    @classmethod
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
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        """Validate that the message is not empty."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for chat with the agent."""
    response: str
    chat_history: list
    success: bool = True
    message: str

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

@app.get("/test-session", response_model=Dict[str, Any])
async def test_session_creation():
    """Test endpoint to verify global session creation works."""
    try:
        # Test creating the global session
        logger.info("🧪 Testing global session creation")
        
        llm, memory = get_or_create_global_chat_session()
        
        # Check if session was created
        if global_chat_session is not None:
            logger.info("✅ Global session created successfully")
            return {
                "success": True,
                "session_type": "global",
                "total_sessions": 1,
                "message": "Global session creation test passed"
            }
        else:
            logger.error("❌ Global session was not created")
            return {
                "success": False,
                "error": "Global session was not stored",
                "message": "Global session creation test failed"
            }
            
    except Exception as e:
        logger.error(f"❌ Error in test session creation: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Global session creation test failed with exception"
        }

@app.post("/debug-chat", response_model=Dict[str, Any])
async def debug_chat_request(request: ChatRequest):
    """Debug endpoint to see what's happening with the ChatRequest validation."""
    logger.info(f"🔍 Debug chat request received")
    logger.info(f"📝 Message: {request.message}")
    logger.info(f"🌐 URL: {request.url}")
    
    return {
        "debug_info": {
            "message": request.message,
            "url": str(request.url) if request.url else None,
            "validation_passed": True
        },
        "message": "Debug request processed successfully"
    }

@app.get("/sessions", response_model=Dict[str, Any])
async def get_sessions_info():
    """Get information about the global chat session."""
    try:
        # Clean up old session first
        cleanup_old_sessions()
        
        session_info = {}
        if global_chat_session is not None:
            session_info["global_session"] = {
                "created_at": global_chat_session["created_at"],
                "last_activity": global_chat_session["last_activity"],
                "age_hours": round((time.time() - global_chat_session["created_at"]) / 3600, 2),
                "idle_hours": round((time.time() - global_chat_session["last_activity"]) / 3600, 2)
            }
        
        return {
            "active_sessions": 1 if global_chat_session is not None else 0,
            "session_type": "global",
            "sessions": session_info
        }
    except Exception as e:
        logger.error(f"❌ Error getting sessions info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sessions info: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete the global chat session."""
    try:
        if session_id == "global_session":  # Only global session can be deleted
            global global_chat_session
            global_chat_session = None
            logger.info("🗑️ Deleted global chat session")
            return {"message": "Global session deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Only 'global_session' can be deleted"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting global session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete global session: {str(e)}"
        )

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for the global session."""
    try:
        if session_id != "global_session":
            raise HTTPException(
                status_code=404,
                detail=f"Only 'global_session' history is available"
            )
        
        if global_chat_session is None:
            raise HTTPException(
                status_code=404,
                detail="Global session not found"
            )
        
        session_data = global_chat_session
        memory = session_data["memory"]
        
        # Get chat history from memory
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        # Format chat history for display
        formatted_history = []
        for i, msg in enumerate(chat_history):
            if hasattr(msg, 'content'):
                formatted_history.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": msg.content,
                    "timestamp": session_data["created_at"] + (i * 60)  # Approximate timestamp
                })
            else:
                formatted_history.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": str(msg),
                    "timestamp": session_data["created_at"] + (i * 60)
                })
        
        return {
            "session_id": "global_session",
            "chat_history": formatted_history,
            "total_messages": len(formatted_history),
            "session_age_hours": round((time.time() - session_data["created_at"]) / 3600, 2),
            "last_activity_hours": round((time.time() - session_data["last_activity"]) / 3600, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting global session history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get global session history: {str(e)}"
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

@app.post("/chat", response_model=ChatResponse)
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
        
        # Get the global chat session (creates one if it doesn't exist)
        logger.info("🔧 Getting global chat session")
        llm, memory = get_or_create_global_chat_session()
        
        # Get response using simple chat function
        url_str = str(request.url) if request.url else None
        logger.info(f"🌐 Processing chat with URL: {url_str}")
        response = chat_with_agent_simple(request.message, url_str, llm, memory)
        
        logger.info("✅ Chat response generated successfully")
        logger.info(f"💬 Response length: {len(response.get('response', ''))}")
        logger.info(f"📝 Chat history length: {len(response.get('chat_history', []))}")
        
        return ChatResponse(
            response=response["response"],
            chat_history=response.get("chat_history", []),
            success=True,
            message="Chat response generated successfully"
        )
        
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
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "message": "Request failed"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
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
    
    # Initialize session management
    logger.info("🧹 Initializing session management...")
    cleanup_old_sessions()
    
    # Start background session cleanup task
    asyncio.create_task(periodic_session_cleanup())
    
    logger.info("✅ AICO Web Summarization API started successfully")

async def periodic_session_cleanup():
    """Periodically clean up old chat sessions."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            cleanup_old_sessions()
            logger.info("🧹 Periodic session cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error in periodic session cleanup: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🛑 Shutting down AICO Web Summarization API...")
    cleanup_old_sessions()  # Clean up sessions on shutdown

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
