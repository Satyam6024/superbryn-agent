"""LLM Service with Gemini (primary) and Groq (fallback) support."""

import json
import logging
from typing import Any, Optional

from .providers import GeminiProvider, GroqProvider, LLMResponse, ProviderType

logger = logging.getLogger(__name__)


# Tool definitions for appointment management
APPOINTMENT_TOOLS = [
    {
        "name": "identify_user",
        "description": "Ask for and record the user's phone number to identify them. Use this when you need to identify a user before booking or retrieving appointments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "The user's phone number in any format"
                }
            },
            "required": ["phone_number"]
        }
    },
    {
        "name": "fetch_slots",
        "description": "Fetch available appointment slots. Use this when the user wants to know what times are available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Optional specific date to check (YYYY-MM-DD format). If not provided, returns next available slots."
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Desired appointment duration in minutes (15, 30, 45, or 60)",
                    "enum": [15, 30, 45, 60]
                }
            },
            "required": []
        }
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment for the user. Requires user to be identified first. Use this when the user confirms they want to book a specific slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Appointment date in YYYY-MM-DD format"
                },
                "time": {
                    "type": "string",
                    "description": "Appointment time in HH:MM format (24-hour)"
                },
                "purpose": {
                    "type": "string",
                    "description": "Purpose or type of appointment"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Appointment duration in minutes",
                    "default": 30
                },
                "user_name": {
                    "type": "string",
                    "description": "User's name for the appointment"
                }
            },
            "required": ["date", "time"]
        }
    },
    {
        "name": "retrieve_appointments",
        "description": "Retrieve the user's appointments. Use this when the user asks about their existing appointments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_past": {
                    "type": "boolean",
                    "description": "Whether to include past appointments",
                    "default": False
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status",
                    "enum": ["scheduled", "cancelled", "completed", "all"]
                }
            },
            "required": []
        }
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel an existing appointment. Use when user wants to cancel a booking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "string",
                    "description": "The ID of the appointment to cancel"
                },
                "date": {
                    "type": "string",
                    "description": "The date of the appointment to cancel (if ID unknown)"
                },
                "time": {
                    "type": "string",
                    "description": "The time of the appointment to cancel (if ID unknown)"
                }
            },
            "required": []
        }
    },
    {
        "name": "modify_appointment",
        "description": "Change the date or time of an existing appointment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "string",
                    "description": "The ID of the appointment to modify"
                },
                "current_date": {
                    "type": "string",
                    "description": "Current date of appointment (if ID unknown)"
                },
                "current_time": {
                    "type": "string",
                    "description": "Current time of appointment (if ID unknown)"
                },
                "new_date": {
                    "type": "string",
                    "description": "New date for the appointment (YYYY-MM-DD)"
                },
                "new_time": {
                    "type": "string",
                    "description": "New time for the appointment (HH:MM)"
                }
            },
            "required": []
        }
    },
    {
        "name": "end_conversation",
        "description": "End the conversation when the user is done or says goodbye. This will generate a summary and close the call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for ending (user_requested, task_completed, timeout)"
                }
            },
            "required": []
        }
    }
]


def get_system_prompt(
    agent_name: str = "Bryn",
    user_context: Optional[str] = None,
    is_returning_user: bool = False,
    user_name: Optional[str] = None,
) -> str:
    """Generate the system prompt for the LLM."""

    greeting_context = ""
    if is_returning_user and user_name:
        greeting_context = f"The user is a returning customer named {user_name}. Greet them warmly by name."
    elif is_returning_user:
        greeting_context = "The user has interacted with us before. Welcome them back."
    else:
        greeting_context = "This appears to be a new user. Give them a friendly introduction."

    return f"""You are {agent_name}, a friendly and professional appointment booking assistant. Your role is to help users book, manage, and retrieve their appointments through natural conversation.

## Your Personality
- Warm but professional
- Clear and concise in responses
- Patient and helpful
- Focused on the task at hand

## Conversation Guidelines
1. {greeting_context}
2. When a user wants to book or manage appointments, you must first identify them by phone number using the identify_user tool.
3. Be conversational but efficient - don't over-explain.
4. When presenting available slots, offer 3-5 options to avoid overwhelming the user.
5. Always confirm details before booking or modifying appointments.
6. If a user asks something unrelated to appointments, politely redirect: "I specialize in appointment scheduling. How can I help you with booking, checking, or managing an appointment?"

## Tool Usage
- Use identify_user when you need the user's phone number (required before booking or retrieving)
- Use fetch_slots to show available times
- Use book_appointment when the user confirms a specific slot
- Use retrieve_appointments to show their existing appointments
- Use cancel_appointment to cancel a booking
- Use modify_appointment to change date/time
- Use end_conversation when the user says goodbye or is done

## Important Rules
- NEVER book without confirming the date, time, and getting user agreement
- ALWAYS identify the user (get phone number) before booking or retrieving their appointments
- If a slot is unavailable, apologize and offer alternatives
- Keep responses concise - this is a voice conversation
- Confirm bookings verbally with all details

{f"Additional context: {user_context}" if user_context else ""}

Remember: You're speaking, not writing. Keep responses natural and conversational."""


class CompatibleResponse:
    """Wrapper to provide Claude-compatible interface for existing code."""

    def __init__(self, llm_response: LLMResponse):
        self._response = llm_response
        self.stop_reason = llm_response.stop_reason
        self.content = self._build_content_blocks()

    def _build_content_blocks(self) -> list:
        """Build Claude-style content blocks."""
        blocks = []

        # Add text block if present
        if self._response.content:
            blocks.append(_TextBlock(self._response.content))

        # Add tool use blocks
        for tc in self._response.tool_calls:
            blocks.append(_ToolUseBlock(
                id=tc.id,
                name=tc.name,
                input=tc.arguments,
            ))

        return blocks


class _TextBlock:
    """Claude-compatible text block."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Claude-compatible tool use block."""

    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class LLMService:
    """LLM Service with Gemini (primary) and Groq (fallback)."""

    def __init__(
        self,
        gemini_api_key: str,
        groq_api_key: str,
        gemini_model: str = "gemini-2.5-flash",
        groq_model: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize LLM service with both providers.

        Args:
            gemini_api_key: Google Gemini API key
            groq_api_key: Groq API key
            gemini_model: Gemini model name
            groq_model: Groq model name
        """
        self.gemini = GeminiProvider(gemini_api_key, gemini_model)
        self.groq = GroqProvider(groq_api_key, groq_model)
        self.tools = APPOINTMENT_TOOLS
        self._last_provider: Optional[ProviderType] = None

        logger.info(
            f"LLM service initialized: Gemini ({gemini_model}) + Groq ({groq_model})"
        )

    def get_tools(self) -> list[dict]:
        """Get tool definitions."""
        return self.tools

    @property
    def last_provider(self) -> Optional[ProviderType]:
        """Get the provider used for the last request."""
        return self._last_provider

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> CompatibleResponse:
        """
        Generate a response with automatic fallback.

        Args:
            messages: Conversation history
            system_prompt: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            Claude-compatible response object
        """
        # Try Gemini first
        try:
            logger.debug("Attempting Gemini request...")
            response = await self.gemini.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                tools=self.tools,
                max_tokens=max_tokens,
            )
            self._last_provider = ProviderType.GEMINI
            logger.debug(f"Gemini response: stop_reason={response.stop_reason}")
            return CompatibleResponse(response)

        except Exception as gemini_error:
            logger.warning(f"Gemini failed, falling back to Groq: {gemini_error}")

            # Fall back to Groq
            try:
                response = await self.groq.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=self.tools,
                    max_tokens=max_tokens,
                )
                self._last_provider = ProviderType.GROQ
                logger.debug(f"Groq response: stop_reason={response.stop_reason}")
                return CompatibleResponse(response)

            except Exception as groq_error:
                logger.error(f"Both providers failed. Gemini: {gemini_error}, Groq: {groq_error}")
                raise RuntimeError(
                    f"All LLM providers failed. Gemini: {gemini_error}, Groq: {groq_error}"
                )

    async def generate_summary(
        self,
        conversation_history: list[dict],
        tool_calls: list[dict],
        appointments_affected: dict,
    ) -> dict:
        """
        Generate a conversation summary.

        Args:
            conversation_history: The conversation messages
            tool_calls: List of tool calls made
            appointments_affected: Dict of booked/modified/cancelled appointments

        Returns:
            Summary dict with text and key points
        """
        summary_prompt = """Analyze this conversation and provide a brief summary. Return a JSON object with:
- "summary": A 1-2 sentence overview of what happened
- "key_points": Array of 3-5 bullet points covering main outcomes
- "preferences": Any preferences the user mentioned (times they prefer, etc.)

Focus on actions taken and outcomes. Be concise. Return ONLY valid JSON, no markdown code blocks."""

        # Build context
        context = f"""Conversation transcript:
{self._format_messages_for_summary(conversation_history)}

Tool calls made: {len(tool_calls)}
Appointments booked: {len(appointments_affected.get('booked', []))}
Appointments modified: {len(appointments_affected.get('modified', []))}
Appointments cancelled: {len(appointments_affected.get('cancelled', []))}"""

        try:
            # Try Gemini first
            try:
                response = await self.gemini.generate_response(
                    messages=[{"role": "user", "content": context}],
                    system_prompt=summary_prompt,
                    tools=None,
                    max_tokens=500,
                )
                text = response.content or ""
            except Exception:
                # Fall back to Groq
                response = await self.groq.generate_response(
                    messages=[{"role": "user", "content": context}],
                    system_prompt=summary_prompt,
                    tools=None,
                    max_tokens=500,
                )
                text = response.content or ""

            # Parse JSON from response
            try:
                # Handle if wrapped in code blocks
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                return json.loads(text.strip())
            except json.JSONDecodeError:
                # Fallback to simple summary
                return {
                    "summary": text[:200],
                    "key_points": ["Conversation completed"],
                    "preferences": {}
                }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": "Conversation completed.",
                "key_points": ["Unable to generate detailed summary"],
                "preferences": {}
            }

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages for summary generation."""
        lines = []
        for msg in messages[-20:]:  # Last 20 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            lines.append(f"{role.upper()}: {content[:200]}")
        return "\n".join(lines)

    def extract_text_response(self, response: CompatibleResponse) -> Optional[str]:
        """Extract text content from response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return None

    def extract_tool_calls(self, response: CompatibleResponse) -> list[dict]:
        """Extract tool use blocks from response."""
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return tool_calls

    async def health_check(self) -> dict[str, bool]:
        """Check health of both providers."""
        return {
            "gemini": await self.gemini.health_check(),
            "groq": await self.groq.health_check(),
        }
