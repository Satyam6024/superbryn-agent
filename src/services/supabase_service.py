"""Supabase service for database operations."""

import logging
from datetime import datetime
from typing import Optional, List
from supabase import create_client, Client

from ..models import Appointment, AppointmentStatus, ConversationSummary, ToolCallLog
from ..models.user import User
from ..models.conversation import EventLog

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for all Supabase database operations."""

    def __init__(self, url: str, key: str):
        """Initialize Supabase client."""
        self.client: Client = create_client(url, key)
        logger.info("Supabase client initialized")

    # ==================== User Operations ====================

    async def get_user_by_phone(self, phone: str) -> Optional[User]:
        """Get user by phone number."""
        try:
            response = self.client.table("users").select("*").eq("phone_number", phone).single().execute()
            if response.data:
                return User(**response.data)
            return None
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None

    async def create_or_update_user(self, phone: str, name: Optional[str] = None) -> User:
        """Create or update user record."""
        try:
            existing = await self.get_user_by_phone(phone)
            now = datetime.utcnow().isoformat()

            if existing:
                update_data = {"last_interaction": now}
                if name:
                    update_data["name"] = name
                self.client.table("users").update(update_data).eq("phone_number", phone).execute()
                existing.last_interaction = datetime.fromisoformat(now)
                if name:
                    existing.name = name
                return existing
            else:
                user_data = {
                    "phone_number": phone,
                    "name": name,
                    "created_at": now,
                    "last_interaction": now,
                    "preferences": {},
                    "total_appointments": 0,
                }
                response = self.client.table("users").insert(user_data).execute()
                return User(**response.data[0])
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            raise

    # ==================== Appointment Operations ====================

    async def get_appointments_by_phone(
        self,
        phone: str,
        status: Optional[AppointmentStatus] = None,
        include_past: bool = True
    ) -> List[Appointment]:
        """Get appointments for a user."""
        try:
            query = self.client.table("appointments").select("*").eq("user_phone", phone)

            if status:
                query = query.eq("status", status.value)

            if not include_past:
                today = datetime.utcnow().strftime("%Y-%m-%d")
                query = query.gte("date", today)

            response = query.order("date", desc=False).order("time", desc=False).execute()
            return [Appointment(**apt) for apt in response.data]
        except Exception as e:
            logger.error(f"Error fetching appointments: {e}")
            return []

    async def get_appointment_by_id(self, appointment_id: str) -> Optional[Appointment]:
        """Get a specific appointment by ID."""
        try:
            response = self.client.table("appointments").select("*").eq("id", appointment_id).single().execute()
            if response.data:
                return Appointment(**response.data)
            return None
        except Exception as e:
            logger.error(f"Error fetching appointment: {e}")
            return None

    async def check_slot_available(self, date: str, time: str) -> bool:
        """Check if a slot is available (not already booked)."""
        try:
            response = (
                self.client.table("appointments")
                .select("id")
                .eq("date", date)
                .eq("time", time)
                .eq("status", AppointmentStatus.SCHEDULED.value)
                .execute()
            )
            return len(response.data) == 0
        except Exception as e:
            logger.error(f"Error checking slot availability: {e}")
            return False

    async def create_appointment(self, appointment: Appointment) -> Appointment:
        """Create a new appointment."""
        try:
            # Check for double booking
            is_available = await self.check_slot_available(appointment.date, appointment.time)
            if not is_available:
                raise ValueError(f"Slot {appointment.date} at {appointment.time} is already booked")

            now = datetime.utcnow().isoformat()
            apt_data = {
                "user_phone": appointment.user_phone,
                "user_name": appointment.user_name,
                "date": appointment.date,
                "time": appointment.time,
                "duration_minutes": appointment.duration_minutes,
                "purpose": appointment.purpose,
                "status": appointment.status,
                "created_at": now,
                "updated_at": now,
                "notes": appointment.notes,
            }

            response = self.client.table("appointments").insert(apt_data).execute()
            created = Appointment(**response.data[0])

            # Update user's appointment count
            await self._increment_user_appointments(appointment.user_phone)

            logger.info(f"Created appointment {created.id} for {appointment.user_phone}")
            return created
        except Exception as e:
            logger.error(f"Error creating appointment: {e}")
            raise

    async def update_appointment(
        self,
        appointment_id: str,
        updates: dict
    ) -> Optional[Appointment]:
        """Update an appointment."""
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            response = (
                self.client.table("appointments")
                .update(updates)
                .eq("id", appointment_id)
                .execute()
            )
            if response.data:
                return Appointment(**response.data[0])
            return None
        except Exception as e:
            logger.error(f"Error updating appointment: {e}")
            raise

    async def cancel_appointment(self, appointment_id: str) -> Optional[Appointment]:
        """Cancel an appointment."""
        return await self.update_appointment(
            appointment_id,
            {"status": AppointmentStatus.CANCELLED.value}
        )

    async def modify_appointment(
        self,
        appointment_id: str,
        new_date: Optional[str] = None,
        new_time: Optional[str] = None
    ) -> Optional[Appointment]:
        """Modify appointment date/time."""
        try:
            # Get current appointment
            current = await self.get_appointment_by_id(appointment_id)
            if not current:
                raise ValueError(f"Appointment {appointment_id} not found")

            target_date = new_date or current.date
            target_time = new_time or current.time

            # Check if new slot is available
            if new_date or new_time:
                is_available = await self.check_slot_available(target_date, target_time)
                if not is_available:
                    raise ValueError(f"Slot {target_date} at {target_time} is not available")

            updates = {}
            if new_date:
                updates["date"] = new_date
            if new_time:
                updates["time"] = new_time

            return await self.update_appointment(appointment_id, updates)
        except Exception as e:
            logger.error(f"Error modifying appointment: {e}")
            raise

    async def _increment_user_appointments(self, phone: str) -> None:
        """Increment user's total appointment count."""
        try:
            user = await self.get_user_by_phone(phone)
            if user:
                self.client.table("users").update(
                    {"total_appointments": user.total_appointments + 1}
                ).eq("phone_number", phone).execute()
        except Exception as e:
            logger.warning(f"Error incrementing appointment count: {e}")

    # ==================== Logging Operations ====================

    async def log_tool_call(self, log: ToolCallLog) -> None:
        """Log a tool call to database."""
        try:
            log_data = {
                "session_id": log.session_id,
                "tool_name": log.tool_name,
                "parameters": log.parameters,
                "result": log.result if isinstance(log.result, (dict, list, str, int, bool, type(None))) else str(log.result),
                "success": log.success,
                "error_message": log.error_message,
                "duration_ms": log.duration_ms,
                "timestamp": log.timestamp.isoformat(),
            }
            self.client.table("tool_call_logs").insert(log_data).execute()
        except Exception as e:
            logger.warning(f"Error logging tool call: {e}")

    async def log_event(self, event: EventLog) -> None:
        """Log a general event."""
        try:
            event_data = {
                "session_id": event.session_id,
                "event_type": event.event_type,
                "event_data": event.event_data,
                "severity": event.severity,
                "timestamp": event.timestamp.isoformat(),
            }
            self.client.table("event_logs").insert(event_data).execute()
        except Exception as e:
            logger.warning(f"Error logging event: {e}")

    async def save_conversation_summary(self, summary: ConversationSummary) -> None:
        """Save conversation summary."""
        try:
            summary_data = {
                "session_id": summary.session_id,
                "user_phone": summary.user_phone,
                "user_name": summary.user_name,
                "summary_text": summary.summary_text,
                "key_points": summary.key_points,
                "appointments_booked": summary.appointments_booked,
                "appointments_modified": summary.appointments_modified,
                "appointments_cancelled": summary.appointments_cancelled,
                "user_preferences": summary.user_preferences,
                "total_turns": summary.total_turns,
                "total_tool_calls": summary.total_tool_calls,
                "duration_seconds": summary.duration_seconds,
                "started_at": summary.started_at.isoformat(),
                "ended_at": summary.ended_at.isoformat(),
            }
            self.client.table("conversation_summaries").insert(summary_data).execute()
        except Exception as e:
            logger.error(f"Error saving conversation summary: {e}")

    async def get_conversation_history(self, phone: str, limit: int = 10) -> List[ConversationSummary]:
        """Get past conversation summaries for a user."""
        try:
            response = (
                self.client.table("conversation_summaries")
                .select("*")
                .eq("user_phone", phone)
                .order("ended_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [ConversationSummary(**s) for s in response.data]
        except Exception as e:
            logger.error(f"Error fetching conversation history: {e}")
            return []

    # ==================== Admin Operations ====================

    async def get_all_appointments(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Appointment]:
        """Get all appointments (for admin panel)."""
        try:
            response = (
                self.client.table("appointments")
                .select("*")
                .order("date", desc=True)
                .order("time", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            return [Appointment(**apt) for apt in response.data]
        except Exception as e:
            logger.error(f"Error fetching all appointments: {e}")
            return []

    async def get_appointments_count(self) -> int:
        """Get total appointment count."""
        try:
            response = self.client.table("appointments").select("id", count="exact").execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Error getting appointment count: {e}")
            return 0
