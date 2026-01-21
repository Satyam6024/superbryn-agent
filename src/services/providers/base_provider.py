"""Base LLM provider abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderType(Enum):
    """Supported LLM provider types."""
    GEMINI = "gemini"
    GROQ = "groq"


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    provider: ProviderType = ProviderType.GEMINI
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        """Get text content or empty string."""
        return self.content or ""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_type: ProviderType

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history in standard format
            system_prompt: System instructions
            tools: Optional list of tools in Anthropic format
            max_tokens: Maximum tokens in response

        Returns:
            Standardized LLMResponse
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass
