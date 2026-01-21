"""Groq LLM provider implementation."""

import json
import logging
from typing import Any, Optional

from groq import AsyncGroq

from .base_provider import BaseLLMProvider, LLMResponse, ProviderType, ToolCall
from ..tool_converter import anthropic_to_groq, convert_messages_to_groq

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider (OpenAI-compatible API)."""

    provider_type = ProviderType.GROQ

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq provider.

        Args:
            api_key: Groq API key
            model: Model name (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key
        self.model = model
        self.client = AsyncGroq(api_key=api_key)

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response using Groq."""
        try:
            # Convert messages to Groq/OpenAI format
            groq_messages = convert_messages_to_groq(messages, system_prompt)

            # Build request parameters
            params = {
                "model": self.model,
                "messages": groq_messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            # Add tools if provided
            if tools:
                groq_tools = anthropic_to_groq(tools)
                params["tools"] = groq_tools
                params["tool_choice"] = "auto"

            # Generate response
            response = await self.client.chat.completions.create(**params)

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Groq response into standardized format."""
        content = None
        tool_calls = []
        stop_reason = "end_turn"

        if response.choices:
            choice = response.choices[0]

            # Check finish reason
            if choice.finish_reason:
                if choice.finish_reason == "stop":
                    stop_reason = "end_turn"
                elif choice.finish_reason == "length":
                    stop_reason = "max_tokens"
                elif choice.finish_reason == "tool_calls":
                    stop_reason = "tool_use"

            # Parse message
            message = choice.message
            if message:
                # Text content
                if message.content:
                    content = message.content

                # Tool calls
                if message.tool_calls:
                    for tc in message.tool_calls:
                        # Parse arguments JSON
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        ))

        if tool_calls:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            provider=ProviderType.GROQ,
            raw_response=response,
        )

    async def health_check(self) -> bool:
        """Check if Groq API is available."""
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
            )
            return response is not None
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            return False
