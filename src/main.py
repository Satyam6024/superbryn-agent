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
from livekit.agents import WorkerOptions, cli

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

# Load environment variables
load_dotenv()


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

    # Create voice agent
    voice_agent = worker.create_voice_agent()

    # Get participant info
    participant_identity = ctx.participant.identity if ctx.participant else "unknown"

    # Determine if returning user (simplified - could enhance with DB lookup)
    is_returning = False
    user_name = None

    # Extract timezone from participant metadata if available
    user_timezone = "UTC"
    if ctx.participant and ctx.participant.metadata:
        try:
            import json
            metadata = json.loads(ctx.participant.metadata)
            user_timezone = metadata.get("timezone", "UTC")
            is_returning = metadata.get("is_returning", False)
            user_name = metadata.get("user_name")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Create session
    session_id = f"{ctx.room.name}-{participant_identity}"
    agent_session = voice_agent.create_agent_session(
        session_id=session_id,
        user_timezone=user_timezone,
        is_returning_user=is_returning,
        user_name=user_name,
    )

    # Start the agent
    await agent_session.start(ctx)


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
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, host="0.0.0.0", port=port)


def main():
    """Main entry point."""
    # Use environment variable for run mode (CLI args conflict with LiveKit CLI)
    run_mode = os.environ.get("RUN_MODE", "both").lower()

    if run_mode == "api":
        logger.info("Starting API server only")
        run_api_server()
    elif run_mode == "agent":
        logger.info("Starting agent worker only")
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        # Default: run both (API in background, agent as main)
        logger.info("Starting both agent worker and API server")
        # Run API server in background thread
        import threading
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()

        # Run agent worker (this blocks)
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
