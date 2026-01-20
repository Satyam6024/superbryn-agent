"""User data models."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class User(BaseModel):
    """Represents a user identified by phone number."""
    phone_number: str = Field(..., description="User's phone number (primary identifier)")
    name: Optional[str] = Field(default=None, description="User's name")
    created_at: Optional[datetime] = Field(default=None)
    last_interaction: Optional[datetime] = Field(default=None)
    preferences: Optional[dict] = Field(default_factory=dict, description="User preferences")
    total_appointments: int = Field(default=0, description="Total appointments made")

    def get_greeting_context(self) -> str:
        """Get context for personalized greeting."""
        if self.name and self.total_appointments > 0:
            return f"returning user {self.name} with {self.total_appointments} previous appointments"
        elif self.name:
            return f"known user {self.name}"
        elif self.total_appointments > 0:
            return f"returning user with {self.total_appointments} previous appointments"
        return "new user"


class ConversationContext(BaseModel):
    """Tracks conversation context for a session."""
    session_id: str = Field(..., description="Unique session identifier")
    user: Optional[User] = Field(default=None, description="Identified user")
    is_identified: bool = Field(default=False)
    turn_count: int = Field(default=0)
    tool_calls: List[str] = Field(default_factory=list)
    pending_action: Optional[str] = Field(default=None)
    mentioned_preferences: dict = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)

    def add_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.tool_calls.append(tool_name)

    def increment_turn(self) -> None:
        """Increment conversation turn count."""
        self.turn_count += 1
