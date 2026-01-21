"""
Configuration settings for SuperBryn Voice Agent.
Loads from environment variables with validation.
"""

from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: Literal["development", "production"] = "development"
    debug: bool = False

    # LiveKit Configuration
    livekit_url: str = Field(..., description="LiveKit server URL")
    livekit_api_key: str = Field(..., description="LiveKit API key")
    livekit_api_secret: str = Field(..., description="LiveKit API secret")

    # Deepgram Configuration (Speech-to-Text)
    deepgram_api_key: str = Field(..., description="Deepgram API key")

    # Cartesia Configuration (Text-to-Speech)
    cartesia_api_key: str = Field(..., description="Cartesia API key")
    cartesia_voice_id: str = Field(
        default="a0e99841-438c-4a64-b679-ae501e7d6091",  # Default neutral voice
        description="Cartesia voice ID for TTS"
    )

    # LLM Configuration (Gemini primary, Groq fallback)
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use"
    )
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use"
    )

    # Beyond Presence Configuration (Avatar)
    beyond_presence_api_key: str = Field(..., description="Beyond Presence API key")
    beyond_presence_avatar_id: str = Field(
        default="default",
        description="Beyond Presence avatar ID"
    )

    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_anon_key: str = Field(..., description="Supabase anonymous key")
    supabase_service_role_key: str = Field(..., description="Supabase service role key")

    # Agent Configuration
    agent_name: str = "Bryn"
    response_timeout_seconds: float = 3.0
    tool_call_timeout_seconds: float = 5.0
    max_conversation_turns: int = 50

    # Appointment Configuration
    slot_duration_minutes: int = 30
    booking_advance_days: int = 30
    business_hours_start: int = 8  # 8 AM
    business_hours_end: int = 20   # 8 PM
    business_days: list[int] = [0, 1, 2, 3, 4, 5]  # Mon-Sat (0=Monday)

    # Admin Configuration
    admin_password: str = Field(default="admin123", description="Admin panel password")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
