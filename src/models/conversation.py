"""Conversation and logging data models."""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class ToolCallLog(BaseModel):
    """Log entry for a tool call."""
    id: Optional[str] = Field(default=None)
    session_id: str = Field(..., description="Session identifier")
    tool_name: str = Field(..., description="Name of the tool called")
    parameters: dict = Field(default_factory=dict, description="Tool parameters")
    result: Optional[Any] = Field(default=None, description="Tool result")
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None, description="Execution time in ms")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_display_dict(self, technical: bool = False) -> dict:
        """Convert to display format for UI."""
        if technical:
            return {
                "tool": self.tool_name,
                "params": self.parameters,
                "result": self.result,
                "success": self.success,
                "duration_ms": self.duration_ms,
                "timestamp": self.timestamp.isoformat(),
            }

        # User-friendly format
        friendly_names = {
            "identify_user": "Identifying user",
            "fetch_slots": "Checking available slots",
            "book_appointment": "Booking appointment",
            "retrieve_appointments": "Fetching appointments",
            "cancel_appointment": "Cancelling appointment",
            "modify_appointment": "Modifying appointment",
            "end_conversation": "Ending conversation",
        }

        friendly_name = friendly_names.get(self.tool_name, self.tool_name)

        return {
            "action": friendly_name,
            "status": "completed" if self.success else "failed",
            "details": self._get_friendly_details(),
            "timestamp": self.timestamp.isoformat(),
        }

    def _get_friendly_details(self) -> str:
        """Get user-friendly details based on tool type."""
        if self.tool_name == "book_appointment" and self.success:
            date = self.parameters.get("date", "")
            time = self.parameters.get("time", "")
            return f"Booked for {date} at {time}"
        elif self.tool_name == "fetch_slots":
            count = len(self.result) if isinstance(self.result, list) else 0
            return f"Found {count} available slots"
        elif self.tool_name == "cancel_appointment" and self.success:
            return "Appointment cancelled"
        elif self.tool_name == "identify_user" and self.success:
            return "User identified"
        elif not self.success:
            return self.error_message or "Action failed"
        return ""


class ConversationSummary(BaseModel):
    """Summary of a completed conversation."""
    id: Optional[str] = Field(default=None)
    session_id: str = Field(..., description="Session identifier")
    user_phone: Optional[str] = Field(default=None, description="User's phone if identified")
    user_name: Optional[str] = Field(default=None)

    # Summary content
    summary_text: str = Field(..., description="Brief text summary")
    key_points: List[str] = Field(default_factory=list, description="Bullet points")
    appointments_booked: List[dict] = Field(default_factory=list)
    appointments_modified: List[dict] = Field(default_factory=list)
    appointments_cancelled: List[dict] = Field(default_factory=list)
    user_preferences: dict = Field(default_factory=dict)

    # Metrics
    total_turns: int = Field(default=0)
    total_tool_calls: int = Field(default=0)
    duration_seconds: int = Field(default=0)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime = Field(default_factory=datetime.utcnow)

    def to_display_dict(self) -> dict:
        """Convert to format for frontend display."""
        return {
            "summary": self.summary_text,
            "keyPoints": self.key_points,
            "appointments": {
                "booked": self.appointments_booked,
                "modified": self.appointments_modified,
                "cancelled": self.appointments_cancelled,
            },
            "preferences": self.user_preferences,
            "metrics": {
                "turns": self.total_turns,
                "toolCalls": self.total_tool_calls,
                "durationSeconds": self.duration_seconds,
            },
            "timestamp": self.ended_at.isoformat(),
        }


class EventLog(BaseModel):
    """General event log for Supabase logging."""
    id: Optional[str] = Field(default=None)
    session_id: str = Field(...)
    event_type: str = Field(..., description="Type of event")
    event_data: dict = Field(default_factory=dict)
    severity: str = Field(default="info")  # info, warning, error
    timestamp: datetime = Field(default_factory=datetime.utcnow)
