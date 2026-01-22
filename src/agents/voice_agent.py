"""
Main Voice Agent implementation using LiveKit Agents framework.
Handles real-time voice conversations with tool calling capabilities.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List, Callable

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    RunContext,
    function_tool,
    llm,
)
from livekit.plugins import deepgram, cartesia

from ..services.supabase_service import SupabaseService
from ..services.llm_service import LLMService, get_system_prompt
from ..services.slot_generator import SlotGenerator
from ..tools.appointment_tools import AppointmentTools, ToolResult, ConversationState
from ..models.conversation import ToolCallLog, ConversationSummary, EventLog

logger = logging.getLogger(__name__)


class VoiceAgent:
    """
    Main voice agent that orchestrates:
    - LiveKit for real-time audio/video
    - Deepgram for speech-to-text
    - Cartesia for text-to-speech
    - Gemini/Groq for conversation intelligence
    - Appointment tools for booking management
    """

    def __init__(
        self,
        supabase_service: SupabaseService,
        llm_service: LLMService,
        slot_generator: SlotGenerator,
        cartesia_api_key: str,
        cartesia_voice_id: str,
        deepgram_api_key: str,
        agent_name: str = "Bryn",
        on_tool_call: Optional[Callable] = None,
        on_transcript: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
    ):
        """
        Initialize the voice agent.

        Args:
            supabase_service: Database service
            llm_service: LLM service (Gemini/Groq)
            slot_generator: Appointment slot generator
            cartesia_api_key: Cartesia API key
            cartesia_voice_id: Cartesia voice ID
            deepgram_api_key: Deepgram API key
            agent_name: Agent's name
            on_tool_call: Callback when tool is called
            on_transcript: Callback for transcript updates
            on_state_change: Callback for state changes
        """
        self.db = supabase_service
        self.llm = llm_service
        self.slot_generator = slot_generator
        self.agent_name = agent_name

        # Initialize tools
        self.tools = AppointmentTools(supabase_service, slot_generator)

        # STT/TTS configuration
        self.deepgram_api_key = deepgram_api_key
        self.cartesia_api_key = cartesia_api_key
        self.cartesia_voice_id = cartesia_voice_id

        # Callbacks for frontend updates
        self.on_tool_call = on_tool_call
        self.on_transcript = on_transcript
        self.on_state_change = on_state_change

        # Session state
        self.session_id: Optional[str] = None
        self.conversation_history: List[dict] = []
        self.tool_call_logs: List[ToolCallLog] = []
        self.started_at: Optional[datetime] = None
        self.is_active = False

    def create_agent_session(
        self,
        session_id: str,
        user_timezone: str = "UTC",
        is_returning_user: bool = False,
        user_name: Optional[str] = None,
    ) -> "BrynAgentSession":
        """
        Create a new agent session for a conversation.

        Args:
            session_id: Unique session identifier
            user_timezone: User's timezone
            is_returning_user: Whether this is a returning user
            user_name: User's name if known

        Returns:
            BrynAgentSession instance
        """
        self.session_id = session_id
        self.started_at = datetime.utcnow()
        self.is_active = True
        self.conversation_history = []
        self.tool_call_logs = []

        # Initialize tools with session
        self.tools.init_conversation(session_id, user_timezone)

        # Get system prompt
        system_prompt = get_system_prompt(
            agent_name=self.agent_name,
            is_returning_user=is_returning_user,
            user_name=user_name,
        )

        return BrynAgentSession(
            voice_agent=self,
            session_id=session_id,
            system_prompt=system_prompt,
        )

    async def handle_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> ToolResult:
        """
        Handle a tool call from the LLM.

        Args:
            tool_name: Name of the tool
            tool_input: Tool parameters

        Returns:
            ToolResult from execution
        """
        start_time = time.time()

        # Execute the tool
        result = await self.tools.execute_tool(tool_name, tool_input)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Create log entry
        log = ToolCallLog(
            session_id=self.session_id,
            tool_name=tool_name,
            parameters=tool_input,
            result=result.data,
            success=result.success,
            error_message=result.error,
            duration_ms=duration_ms,
        )
        self.tool_call_logs.append(log)

        # Log to database
        await self.db.log_tool_call(log)

        # Notify frontend
        if self.on_tool_call:
            try:
                await self.on_tool_call(log.to_display_dict(technical=False))
            except Exception as e:
                logger.warning(f"Error in tool call callback: {e}")

        logger.info(f"Tool {tool_name} executed: success={result.success}, duration={duration_ms}ms")

        return result

    async def generate_summary(self) -> ConversationSummary:
        """Generate conversation summary at end of call."""
        try:
            # Get appointments affected
            state = self.tools.state
            appointments_affected = {
                "booked": state.appointments_booked if state else [],
                "modified": state.appointments_modified if state else [],
                "cancelled": state.appointments_cancelled if state else [],
            }

            # Generate summary via LLM
            summary_data = await self.llm.generate_summary(
                self.conversation_history,
                [log.to_display_dict(technical=True) for log in self.tool_call_logs],
                appointments_affected,
            )

            # Calculate duration
            duration = 0
            if self.started_at:
                duration = int((datetime.utcnow() - self.started_at).total_seconds())

            # Create summary object
            summary = ConversationSummary(
                session_id=self.session_id,
                user_phone=state.user_phone if state else None,
                user_name=state.user_name if state else None,
                summary_text=summary_data.get("summary", "Conversation completed."),
                key_points=summary_data.get("key_points", []),
                appointments_booked=appointments_affected["booked"],
                appointments_modified=appointments_affected["modified"],
                appointments_cancelled=appointments_affected["cancelled"],
                user_preferences=summary_data.get("preferences", {}),
                total_turns=len(self.conversation_history) // 2,
                total_tool_calls=len(self.tool_call_logs),
                duration_seconds=duration,
                started_at=self.started_at or datetime.utcnow(),
                ended_at=datetime.utcnow(),
            )

            # Save to database
            await self.db.save_conversation_summary(summary)

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ConversationSummary(
                session_id=self.session_id,
                summary_text="Conversation ended.",
                key_points=["Summary generation failed"],
            )

    async def end_conversation(self) -> ConversationSummary:
        """End the conversation and generate summary."""
        self.is_active = False

        # Log end event
        await self.db.log_event(EventLog(
            session_id=self.session_id,
            event_type="conversation_ended",
            event_data={"reason": "normal_end"},
        ))

        # Generate and return summary
        return await self.generate_summary()


class BrynAgentSession(Agent):
    """
    LiveKit Agent session implementation.
    Handles the real-time conversation flow.
    """

    def __init__(
        self,
        voice_agent: VoiceAgent,
        session_id: str,
        system_prompt: str,
    ):
        """Initialize agent session."""
        super().__init__(instructions=system_prompt)
        self.voice_agent = voice_agent
        self.session_id = session_id
        self._system_prompt = system_prompt

        # Configure STT (use different name to avoid Agent property conflict)
        self._stt_instance = deepgram.STT(
            api_key=voice_agent.deepgram_api_key,
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
        )

        # Configure TTS (use different name to avoid Agent property conflict)
        self._tts_instance = cartesia.TTS(
            api_key=voice_agent.cartesia_api_key,
            voice=voice_agent.cartesia_voice_id,
        )

    @function_tool()
    async def identify_user(self, phone_number: str) -> str:
        """Ask for and record the user's phone number."""
        result = await self.voice_agent.handle_tool_call(
            "identify_user",
            {"phone_number": phone_number}
        )
        return result.verbal_response

    @function_tool()
    async def fetch_slots(
        self,
        date: Optional[str] = None,
        duration_minutes: Optional[int] = None,
    ) -> str:
        """Fetch available appointment slots."""
        params = {}
        if date:
            params["date"] = date
        if duration_minutes:
            params["duration_minutes"] = duration_minutes

        result = await self.voice_agent.handle_tool_call("fetch_slots", params)
        return result.verbal_response

    @function_tool()
    async def book_appointment(
        self,
        date: str,
        time: str,
        purpose: Optional[str] = None,
        duration_minutes: int = 30,
        user_name: Optional[str] = None,
    ) -> str:
        """Book an appointment for the user."""
        params = {
            "date": date,
            "time": time,
            "duration_minutes": duration_minutes,
        }
        if purpose:
            params["purpose"] = purpose
        if user_name:
            params["user_name"] = user_name

        result = await self.voice_agent.handle_tool_call("book_appointment", params)
        return result.verbal_response

    @function_tool()
    async def retrieve_appointments(
        self,
        include_past: bool = False,
        status: Optional[str] = None,
    ) -> str:
        """Retrieve the user's appointments."""
        params = {"include_past": include_past}
        if status:
            params["status"] = status

        result = await self.voice_agent.handle_tool_call("retrieve_appointments", params)
        return result.verbal_response

    @function_tool()
    async def cancel_appointment(
        self,
        appointment_id: Optional[str] = None,
        date: Optional[str] = None,
        time: Optional[str] = None,
    ) -> str:
        """Cancel an existing appointment."""
        params = {}
        if appointment_id:
            params["appointment_id"] = appointment_id
        if date:
            params["date"] = date
        if time:
            params["time"] = time

        result = await self.voice_agent.handle_tool_call("cancel_appointment", params)
        return result.verbal_response

    @function_tool()
    async def modify_appointment(
        self,
        appointment_id: Optional[str] = None,
        current_date: Optional[str] = None,
        current_time: Optional[str] = None,
        new_date: Optional[str] = None,
        new_time: Optional[str] = None,
    ) -> str:
        """Change the date or time of an existing appointment."""
        params = {}
        if appointment_id:
            params["appointment_id"] = appointment_id
        if current_date:
            params["current_date"] = current_date
        if current_time:
            params["current_time"] = current_time
        if new_date:
            params["new_date"] = new_date
        if new_time:
            params["new_time"] = new_time

        result = await self.voice_agent.handle_tool_call("modify_appointment", params)
        return result.verbal_response

    @function_tool()
    async def end_conversation(self, reason: str = "user_requested") -> str:
        """End the conversation when the user is done."""
        result = await self.voice_agent.handle_tool_call(
            "end_conversation",
            {"reason": reason}
        )
        return result.verbal_response

    async def on_enter(self) -> str:
        """Called when agent enters the room. Return greeting message."""
        logger.info(f"Agent entered room for session {self.session_id}")

        # Return initial greeting (will be spoken by the agent)
        return self._get_greeting()

    def _get_greeting(self) -> str:
        """Get appropriate greeting based on context."""
        state = self.voice_agent.tools.state
        if state and state.user_name:
            return f"Hi {state.user_name}! Welcome back. How can I help you today?"
        elif state and state.is_identified:
            return "Welcome back! How can I help you today?"
        else:
            return f"Hi, I'm {self.voice_agent.agent_name}! I can help you book, check, or manage your appointments. How can I assist you today?"

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        """Called when user finishes speaking."""
        # Log transcript
        if self.voice_agent.on_transcript:
            try:
                await self.voice_agent.on_transcript({
                    "role": "user",
                    "content": new_message.content,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Error in transcript callback: {e}")

        # Add to history
        self.voice_agent.conversation_history.append({
            "role": "user",
            "content": new_message.content,
        })

    async def on_agent_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        """Called when agent finishes speaking."""
        # Log transcript
        if self.voice_agent.on_transcript:
            try:
                await self.voice_agent.on_transcript({
                    "role": "assistant",
                    "content": new_message.content,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Error in transcript callback: {e}")

        # Add to history
        self.voice_agent.conversation_history.append({
            "role": "assistant",
            "content": new_message.content,
        })

        # Check if conversation should end
        state = self.voice_agent.tools.state
        if state and state.should_end:
            await self.voice_agent.end_conversation()

    async def on_close(self) -> None:
        """Called when session closes."""
        logger.info(f"Session {self.session_id} closing")

        if self.voice_agent.is_active:
            await self.voice_agent.end_conversation()
