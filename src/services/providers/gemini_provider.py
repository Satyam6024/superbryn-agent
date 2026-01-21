"""Google Gemini LLM provider implementation."""

import json
import logging
from typing import Any, Optional

from google import genai
from google.genai import types

from .base_provider import BaseLLMProvider, LLMResponse, ProviderType, ToolCall
from ..tool_converter import anthropic_to_gemini, convert_messages_to_gemini

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    provider_type = ProviderType.GEMINI

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            model: Model name (default: gemini-2.5-flash)
        """
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        try:
            # Convert messages to Gemini format
            system_instruction, gemini_messages = convert_messages_to_gemini(
                messages, system_prompt
            )

            # Build generation config
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=0.7,
            )

            # Convert and add tools if provided
            if tools:
                gemini_tools = anthropic_to_gemini(tools)
                config.tools = [types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t.get("parameters"),
                    )
                    for t in gemini_tools
                ])]
                # Allow both text and function calls
                config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO"
                    )
                )

            # Generate response
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=config,
            )

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Gemini response into standardized format."""
        content = None
        tool_calls = []
        stop_reason = "end_turn"

        if response.candidates:
            candidate = response.candidates[0]

            # Check finish reason
            if candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)
                if "STOP" in finish_reason:
                    stop_reason = "end_turn"
                elif "MAX_TOKENS" in finish_reason:
                    stop_reason = "max_tokens"
                elif "SAFETY" in finish_reason:
                    stop_reason = "safety"

            # Parse parts
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Text content
                    if hasattr(part, "text") and part.text:
                        content = part.text

                    # Function call
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        # Convert args to dict
                        args = {}
                        if fc.args:
                            # fc.args is a google.protobuf.struct_pb2.Struct
                            for key, value in fc.args.items():
                                args[key] = value

                        tool_calls.append(ToolCall(
                            id=f"call_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=args,
                        ))

        if tool_calls:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            provider=ProviderType.GEMINI,
            raw_response=response,
        )

    async def health_check(self) -> bool:
        """Check if Gemini API is available."""
        try:
            # Simple test request
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            return response is not None
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False
