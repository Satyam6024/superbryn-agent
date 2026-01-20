"""
Beyond Presence avatar integration service.
Handles avatar session management and audio synchronization.
"""

import logging
import asyncio
from typing import Optional, Callable
import httpx

logger = logging.getLogger(__name__)


class BeyondPresenceService:
    """
    Service for Beyond Presence avatar integration.

    Beyond Presence provides real-time avatar rendering that syncs
    with audio output for natural lip-sync and expressions.
    """

    BASE_URL = "https://api.beyondpresence.ai/v1"

    def __init__(self, api_key: str, avatar_id: str = "default"):
        """
        Initialize Beyond Presence service.

        Args:
            api_key: Beyond Presence API key
            avatar_id: Avatar ID to use (default uses their demo avatar)
        """
        self.api_key = api_key
        self.avatar_id = avatar_id
        self.session_id: Optional[str] = None
        self.stream_url: Optional[str] = None
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def create_session(
        self,
        room_name: str,
        audio_input_url: Optional[str] = None,
    ) -> dict:
        """
        Create a new avatar session.

        Args:
            room_name: LiveKit room name for the session
            audio_input_url: URL for audio input stream

        Returns:
            Session configuration with stream URL
        """
        try:
            payload = {
                "avatar_id": self.avatar_id,
                "room_name": room_name,
                "settings": {
                    "resolution": "720p",
                    "fps": 30,
                    "quality": "high",
                    "background": "transparent",
                },
            }

            if audio_input_url:
                payload["audio_input_url"] = audio_input_url

            response = await self._client.post(
                f"{self.BASE_URL}/sessions",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            self.session_id = data.get("session_id")
            self.stream_url = data.get("stream_url")

            logger.info(f"Created Beyond Presence session: {self.session_id}")

            return {
                "session_id": self.session_id,
                "stream_url": self.stream_url,
                "avatar_id": self.avatar_id,
            }

        except httpx.HTTPError as e:
            logger.error(f"Failed to create Beyond Presence session: {e}")
            # Return fallback for demo purposes
            return {
                "session_id": f"demo-{room_name}",
                "stream_url": None,
                "avatar_id": self.avatar_id,
                "error": str(e),
            }

    async def get_stream_url(self) -> Optional[str]:
        """Get the current avatar stream URL."""
        return self.stream_url

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Send audio chunk for lip-sync processing.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Success status
        """
        if not self.session_id:
            logger.warning("No active session for audio chunk")
            return False

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/sessions/{self.session_id}/audio",
                content=audio_data,
                headers={"Content-Type": "audio/pcm"},
            )
            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to send audio chunk: {e}")
            return False

    async def set_expression(self, expression: str) -> bool:
        """
        Set avatar expression/emotion.

        Args:
            expression: Expression name (neutral, happy, thinking, etc.)

        Returns:
            Success status
        """
        if not self.session_id:
            return False

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/sessions/{self.session_id}/expression",
                json={"expression": expression},
            )
            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to set expression: {e}")
            return False

    async def set_state(self, state: str) -> bool:
        """
        Set avatar state (idle, listening, speaking, thinking).

        Args:
            state: State name

        Returns:
            Success status
        """
        if not self.session_id:
            return False

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/sessions/{self.session_id}/state",
                json={"state": state},
            )
            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to set state: {e}")
            return False

    async def end_session(self) -> bool:
        """End the current avatar session."""
        if not self.session_id:
            return True

        try:
            response = await self._client.delete(
                f"{self.BASE_URL}/sessions/{self.session_id}",
            )
            response.raise_for_status()

            logger.info(f"Ended Beyond Presence session: {self.session_id}")
            self.session_id = None
            self.stream_url = None
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to end session: {e}")
            return False

    async def get_available_avatars(self) -> list:
        """Get list of available avatars."""
        try:
            response = await self._client.get(f"{self.BASE_URL}/avatars")
            response.raise_for_status()
            return response.json().get("avatars", [])

        except httpx.HTTPError as e:
            logger.error(f"Failed to get avatars: {e}")
            return []

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AvatarStateManager:
    """
    Manages avatar state transitions based on conversation flow.
    Ensures smooth transitions between states.
    """

    STATES = {
        "idle": "idle",
        "listening": "listening",
        "thinking": "thinking",
        "speaking": "speaking",
    }

    def __init__(self, beyond_presence: BeyondPresenceService):
        """Initialize state manager."""
        self.bp = beyond_presence
        self.current_state = "idle"
        self._transition_lock = asyncio.Lock()

    async def transition_to(self, new_state: str) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state

        Returns:
            Success status
        """
        if new_state not in self.STATES:
            logger.warning(f"Invalid state: {new_state}")
            return False

        if new_state == self.current_state:
            return True

        async with self._transition_lock:
            success = await self.bp.set_state(self.STATES[new_state])
            if success:
                self.current_state = new_state
                logger.debug(f"Avatar state: {new_state}")
            return success

    async def on_user_speaking(self):
        """Called when user starts speaking."""
        await self.transition_to("listening")

    async def on_user_stopped(self):
        """Called when user stops speaking."""
        await self.transition_to("thinking")

    async def on_agent_speaking(self):
        """Called when agent starts speaking."""
        await self.transition_to("speaking")

    async def on_agent_stopped(self):
        """Called when agent stops speaking."""
        await self.transition_to("idle")

    async def on_tool_call(self):
        """Called when a tool is being executed."""
        await self.transition_to("thinking")
