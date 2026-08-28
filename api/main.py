"""
TPQ AI Assistant - FastAPI Application

Main application entry point. Handles:
- Model loading at startup (loaded once, kept in memory)
- Route registration
- CORS middleware
- Static file serving for frontend
- Health check endpoint
- Global exception handling

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global model storage (loaded once at startup)
_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """Get the loaded model and tokenizer. Raises if not loaded."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model is not loaded yet. Please wait for startup to complete.")
    return _model, _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the model at startup and cleans up at shutdown.
    """
    global _model, _tokenizer

    print("=" * 60)
    print("  TPQ AI Assistant - Starting API Server")
    print("=" * 60)
    print()

    # Load model at startup
    print("Loading model...")
    try:
        from inference.model_loader import load_model

        _model, _tokenizer = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("API will start but /api/chat will return 503 errors.")
        print("Make sure the model is trained first: python training/train.py")

    print()
    print("API is ready!")
    print("  Docs: http://localhost:8000/docs")
    print("  Chat: http://localhost:8000")
    print()

    yield  # Application runs here

    # Cleanup at shutdown
    print("Shutting down... Cleaning up model resources.")
    _model = None
    _tokenizer = None


# ================================================================
# FastAPI Application
# ================================================================

app = FastAPI(
    title="TPQ AI Assistant",
    description=(
        "REST API for TPQ AI Assistant — a chatbot powered by "
        "fine-tuned Qwen2.5 (SFT + LoRA via Unsloth) for answering "
        "questions about TPQ (Taman Pendidikan Al-Quran) administration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ================================================================
# CORS Middleware
# ================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Static Files (Frontend)
# ================================================================

frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
)

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# ================================================================
# Routes
# ================================================================

from api.routes.chat import router as chat_router
from api.schemas.chat import HealthResponse

app.include_router(chat_router, tags=["Chat"])


@app.get(
    "/",
    summary="Serve frontend",
    description="Serves the web chatbot interface.",
    include_in_schema=False,
)
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "TPQ AI Assistant API. Visit /docs for API documentation."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and model are ready.",
    tags=["System"],
)
async def health_check():
    """Return the health status of the API."""
    model_status = "qwen-tpq-sft" if _model is not None else "not loaded"
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model=model_status,
    )


# ================================================================
# Global Exception Handler
# ================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions globally."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again later.",
        },
    )
