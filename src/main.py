"""
Main entry point for SuperBryn Voice Agent.
Starts the LiveKit agent worker and HTTP API server.
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from aiohttp import web
from dotenv import load_dotenv
from livekit.agents import WorkerOptions, cli, AgentSession
from livekit.plugins import google as google_llm
from livekit.plugins import silero

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.services.supabase_service import SupabaseService
from src.services.llm_service import LLMService
from src.services.slot_generator import SlotGenerator
from src.agents.voice_agent import VoiceAgent, BrynAgentSession
from src.api.routes import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables (override=True ensures .env values take precedence)
load_dotenv(override=True)


class AgentWorker:
    """Worker that manages voice agent instances."""

    def __init__(self):
        """Initialize the worker with services."""
        self.settings = get_settings()

        # Initialize services
        self.supabase = SupabaseService(
            url=self.settings.supabase_url,
            key=self.settings.supabase_service_role_key,
        )

        self.llm = LLMService(
            gemini_api_key=self.settings.gemini_api_key,
            groq_api_key=self.settings.groq_api_key,
            gemini_model=self.settings.gemini_model,
            groq_model=self.settings.groq_model,
        )

        self.slot_generator = SlotGenerator(
            business_hours_start=self.settings.business_hours_start,
            business_hours_end=self.settings.business_hours_end,
            business_days=self.settings.business_days,
            booking_advance_days=self.settings.booking_advance_days,
            default_slot_duration=self.settings.slot_duration_minutes,
        )

        logger.info("AgentWorker initialized with all services")

    def create_voice_agent(self) -> VoiceAgent:
        """Create a new voice agent instance."""
        return VoiceAgent(
            supabase_service=self.supabase,
            llm_service=self.llm,
            slot_generator=self.slot_generator,
            cartesia_api_key=self.settings.cartesia_api_key,
            cartesia_voice_id=self.settings.cartesia_voice_id,
            deepgram_api_key=self.settings.deepgram_api_key,
            agent_name=self.settings.agent_name,
        )


# Global worker instance
worker: Optional[AgentWorker] = None


async def entrypoint(ctx):
    """
    LiveKit agent entrypoint.
    Called when a new room is created or participant joins.
    """
    global worker

    if worker is None:
        worker = AgentWorker()

    logger.info(f"Agent entrypoint called for room: {ctx.room.name}")

    # Connect to the room first
    await ctx.connect()

    # Create voice agent
    voice_agent = worker.create_voice_agent()

    # Get first remote participant (the user)
    participant_identity = "unknown"
    user_timezone = "UTC"
    is_returning = False
    user_name = None

    # Check remote participants for metadata
    for p in ctx.room.remote_participants.values():
        participant_identity = p.identity
        if p.metadata:
            try:
                import json
                metadata = json.loads(p.metadata)
                user_timezone = metadata.get("timezone", "UTC")
                is_returning = metadata.get("is_returning", False)
                user_name = metadata.get("user_name")
            except (json.JSONDecodeError, AttributeError):
                pass
        break

    # Create session
    session_id = f"{ctx.room.name}-{participant_identity}"
    bryn_agent = voice_agent.create_agent_session(
        session_id=session_id,
        user_timezone=user_timezone,
        is_returning_user=is_returning,
        user_name=user_name,
    )

    # Create Gemini LLM for the session
    gemini_llm = google_llm.LLM(
        model=worker.settings.gemini_model,
        api_key=worker.settings.gemini_api_key,
    )

    # Create AgentSession with VAD for proper turn detection
    session = AgentSession(
        stt=bryn_agent._stt_instance,
        tts=bryn_agent._tts_instance,
        llm=gemini_llm,
        vad=silero.VAD.load(),
    )

    # Start session with agent and room as keyword arguments
    await session.start(agent=bryn_agent, room=ctx.room)

    logger.info("Session started, saying greeting...")

    # Use say() for direct TTS output
    try:
        greeting = "Hello! I'm Bryn, your voice assistant. I can help you book, check, or manage appointments. How can I help you today?"
        await session.say(greeting)
        logger.info("Greeting spoken successfully")
    except Exception as e:
        logger.error(f"Failed to speak greeting: {e}")


def run_api_server():
    """Run the HTTP API server."""
    settings = get_settings()

    # Initialize services for API
    supabase = SupabaseService(
        url=settings.supabase_url,
        key=settings.supabase_service_role_key,
    )

    # Create app
    app = create_app(
        supabase_service=supabase,
        livekit_api_key=settings.livekit_api_key,
        livekit_api_secret=settings.livekit_api_secret,
        admin_password=settings.admin_password,
    )

    # Run server
    port = int(os.environ.get("PORT", 8082))
    host = os.environ.get("HOST", "0.0.0.0")
    web.run_app(app, host=host, port=port)


def run_api_in_thread():
    """Run the API server in a separate thread (for 'both' mode)."""
    import threading

    def _start_api():
        run_api_server()

    api_thread = threading.Thread(target=_start_api, daemon=True)
    api_thread.start()
    logger.info("API server thread started")


def main():
    """Main entry point."""
    # Use environment variable for run mode (CLI args conflict with LiveKit CLI)
    run_mode = os.environ.get("RUN_MODE", "api").strip().lower()

    if run_mode == "agent":
        logger.info("Starting agent worker only")
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    elif run_mode == "both":
        logger.info("Starting both agent worker and API server")
        # Start API server in a background thread, then run agent worker
        run_api_in_thread()
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        # Default: run API server only
        logger.info("Starting API server")
        run_api_server()


if __name__ == "__main__":
    main()
