"""Appointment data models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class AppointmentStatus(str, Enum):
    """Possible appointment statuses."""
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class TimeSlot(BaseModel):
    """Represents an available time slot."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time: str = Field(..., description="Time in HH:MM format (24-hour)")
    duration_minutes: int = Field(default=30, description="Slot duration in minutes")
    is_available: bool = Field(default=True, description="Whether slot is available")

    @property
    def datetime_str(self) -> str:
        """Get formatted datetime string."""
        return f"{self.date} at {self.time}"


class Appointment(BaseModel):
    """Represents a booked appointment."""
    id: Optional[str] = Field(default=None, description="Unique appointment ID")
    user_phone: str = Field(..., description="User's phone number (identifier)")
    user_name: Optional[str] = Field(default=None, description="User's name")
    date: str = Field(..., description="Appointment date (YYYY-MM-DD)")
    time: str = Field(..., description="Appointment time (HH:MM)")
    duration_minutes: int = Field(default=30, description="Duration in minutes")
    purpose: Optional[str] = Field(default=None, description="Appointment purpose/type")
    status: AppointmentStatus = Field(default=AppointmentStatus.SCHEDULED)
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @property
    def datetime_str(self) -> str:
        """Get formatted datetime string."""
        return f"{self.date} at {self.time}"

    def to_verbal_summary(self) -> str:
        """Generate a verbal summary for TTS."""
        status_text = ""
        if self.status == AppointmentStatus.CANCELLED:
            status_text = " (cancelled)"
        elif self.status == AppointmentStatus.COMPLETED:
            status_text = " (completed)"

        purpose_text = f" for {self.purpose}" if self.purpose else ""
        return f"Appointment on {self.date} at {self.time}{purpose_text}{status_text}"

    class Config:
        use_enum_values = True
