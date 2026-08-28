"""
TPQ AI Assistant - API Schemas

Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question about TPQ administration.",
        examples=["Bagaimana cara melihat nilai anak?"],
    )


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    response: str = Field(
        ...,
        description="The AI assistant's response.",
        examples=[
            "Wali santri dapat melihat nilai anak melalui menu Nilai setelah login ke sistem."
        ],
    )


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(
        ...,
        description="Health status of the API.",
        examples=["healthy"],
    )
    model: str = Field(
        ...,
        description="Name/identifier of the loaded model.",
        examples=["qwen-tpq-sft"],
    )
