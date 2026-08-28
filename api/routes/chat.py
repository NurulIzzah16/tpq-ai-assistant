"""
TPQ AI Assistant - Chat Route

Handles the POST /api/chat endpoint for chat interactions.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Chat with TPQ AI Assistant",
    description="Send a message and receive a response from the TPQ AI Assistant.",
)
async def chat(request: ChatRequest):
    """
    Process a chat message and return AI response.

    The model is loaded once at application startup and kept in memory.
    This endpoint accesses the model via the app state.
    """
    from api.main import get_model_and_tokenizer
    from inference.model_loader import generate_response

    try:
        model, tokenizer = get_model_and_tokenizer()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}",
        )

    try:
        response_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_message=request.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}",
        )

    return ChatResponse(response=response_text)
