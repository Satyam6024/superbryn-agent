"""Appointment management tools for the voice agent."""

import logging
import re
from datetime import datetime
from typing import Optional, List, Any
from dataclasses import dataclass, field

from ..models import Appointment, AppointmentStatus, TimeSlot
from ..models.user import User, ConversationContext
from ..services.supabase_service import SupabaseService
from ..services.slot_generator import SlotGenerator

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    message: str = ""
    verbal_response: str = ""
    error: Optional[str] = None


@dataclass
class ConversationState:
    """Tracks the current conversation state."""
    session_id: str
    user_phone: Optional[str] = None
    user_name: Optional[str] = None
    user: Optional[User] = None
    is_identified: bool = False
    appointments_booked: List[dict] = field(default_factory=list)
    appointments_modified: List[dict] = field(default_factory=list)
    appointments_cancelled: List[dict] = field(default_factory=list)
    mentioned_preferences: dict = field(default_factory=dict)
    user_timezone: str = "UTC"
    should_end: bool = False


class AppointmentTools:
    """
    Tool implementations for appointment management.

    All tools return ToolResult with:
    - success: Whether the operation succeeded
    - data: Structured data for logging/display
    - message: Technical message
    - verbal_response: What to say to the user
    - error: Error message if failed
    """

    def __init__(
        self,
        supabase_service: SupabaseService,
        slot_generator: SlotGenerator,
    ):
        """
        Initialize appointment tools.

        Args:
            supabase_service: Database service
            slot_generator: Slot generation service
        """
        self.db = supabase_service
        self.slots = slot_generator
        self.state: Optional[ConversationState] = None

    def init_conversation(self, session_id: str, timezone: str = "UTC") -> None:
        """Initialize a new conversation state."""
        self.state = ConversationState(
            session_id=session_id,
            user_timezone=timezone,
        )
        logger.info(f"Initialized conversation state for session {session_id}")

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to consistent format."""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)

        # Handle US numbers
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        elif len(digits) > 10:
            return f"+{digits}"

        return digits

    async def identify_user(self, phone_number: str) -> ToolResult:
        """
        Identify user by phone number.

        Args:
            phone_number: User's phone number

        Returns:
            ToolResult with user info
        """
        if not self.state:
            return ToolResult(
                success=False,
                error="Conversation not initialized",
                verbal_response="I'm sorry, there was a technical issue. Please try again."
            )

        try:
            normalized_phone = self._normalize_phone(phone_number)
            logger.info(f"Identifying user with phone: {normalized_phone}")

            # Try to find existing user
            user = await self.db.get_user_by_phone(normalized_phone)

            if user:
                # Returning user
                self.state.user = user
                self.state.user_phone = normalized_phone
                self.state.user_name = user.name
                self.state.is_identified = True

                greeting = f"Welcome back"
                if user.name:
                    greeting += f", {user.name}"
                greeting += "!"

                if user.total_appointments > 0:
                    greeting += f" I can see you have {user.total_appointments} appointment{'s' if user.total_appointments > 1 else ''} in our system."

                return ToolResult(
                    success=True,
                    data={"user": user.model_dump(), "is_returning": True},
                    message=f"User identified: {normalized_phone}",
                    verbal_response=greeting
                )
            else:
                # New user - create record
                user = await self.db.create_or_update_user(normalized_phone)
                self.state.user = user
                self.state.user_phone = normalized_phone
                self.state.is_identified = True

                return ToolResult(
                    success=True,
                    data={"user": user.model_dump(), "is_returning": False},
                    message=f"New user created: {normalized_phone}",
                    verbal_response="Great, I've got your number. How can I help you today?"
                )

        except Exception as e:
            logger.error(f"Error identifying user: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble processing that phone number. Could you please repeat it?"
            )

    async def fetch_slots(
        self,
        date: Optional[str] = None,
        duration_minutes: Optional[int] = None,
    ) -> ToolResult:
        """
        Fetch available appointment slots.

        Args:
            date: Optional specific date to check
            duration_minutes: Desired appointment duration

        Returns:
            ToolResult with available slots
        """
        if not self.state:
            return ToolResult(
                success=False,
                error="Conversation not initialized",
                verbal_response="I'm sorry, there was a technical issue."
            )

        try:
            # Get already booked slots from database
            booked_response = await self._get_booked_slots()

            if date:
                # Get slots for specific date
                available = self.slots.get_slots_for_date(
                    date,
                    user_timezone=self.state.user_timezone,
                    booked_slots=booked_response,
                )
            else:
                # Get next available slots
                available = self.slots.get_available_slots(
                    user_timezone=self.state.user_timezone,
                    duration_minutes=duration_minutes,
                    booked_slots=booked_response,
                    limit=10,
                )

            if not available:
                if date:
                    return ToolResult(
                        success=True,
                        data={"slots": [], "date": date},
                        message="No slots available for specified date",
                        verbal_response=f"I'm sorry, there are no available slots on {date}. Would you like to check another date?"
                    )
                else:
                    return ToolResult(
                        success=True,
                        data={"slots": []},
                        message="No slots available",
                        verbal_response="I'm sorry, we don't have any available slots at the moment. Please try again later."
                    )

            # Format for verbal response
            verbal = self.slots.format_slots_for_speech(available, max_slots=5)

            return ToolResult(
                success=True,
                data={"slots": [s.model_dump() for s in available]},
                message=f"Found {len(available)} available slots",
                verbal_response=f"I have some availability for you. {verbal}. Which time works best?"
            )

        except Exception as e:
            logger.error(f"Error fetching slots: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble checking availability. Please try again."
            )

    async def _get_booked_slots(self) -> List[tuple]:
        """Get list of already booked slots."""
        try:
            # Query all scheduled appointments
            response = self.db.client.table("appointments").select(
                "date, time"
            ).eq("status", "scheduled").execute()

            return [(apt["date"], apt["time"]) for apt in response.data]
        except Exception as e:
            logger.warning(f"Error getting booked slots: {e}")
            return []

    async def book_appointment(
        self,
        date: str,
        time: str,
        purpose: Optional[str] = None,
        duration_minutes: int = 30,
        user_name: Optional[str] = None,
    ) -> ToolResult:
        """
        Book an appointment.

        Args:
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            purpose: Purpose of appointment
            duration_minutes: Duration in minutes
            user_name: User's name

        Returns:
            ToolResult with booking confirmation
        """
        if not self.state:
            return ToolResult(
                success=False,
                error="Conversation not initialized",
                verbal_response="I'm sorry, there was a technical issue."
            )

        if not self.state.is_identified:
            return ToolResult(
                success=False,
                error="User not identified",
                verbal_response="Before I can book an appointment, I'll need your phone number. What's your phone number?"
            )

        try:
            # Validate the slot
            is_valid, error_msg = self.slots.validate_slot(
                date, time, self.state.user_timezone
            )
            if not is_valid:
                return ToolResult(
                    success=False,
                    error=error_msg,
                    verbal_response=error_msg
                )

            # Check if slot is still available
            is_available = await self.db.check_slot_available(date, time)
            if not is_available:
                return ToolResult(
                    success=False,
                    error="Slot already booked",
                    verbal_response="I'm sorry, that slot was just taken. Would you like me to find another available time?"
                )

            # Update user name if provided
            if user_name and not self.state.user_name:
                self.state.user_name = user_name
                await self.db.create_or_update_user(
                    self.state.user_phone,
                    name=user_name
                )

            # Create appointment
            appointment = Appointment(
                user_phone=self.state.user_phone,
                user_name=self.state.user_name or user_name,
                date=date,
                time=time,
                duration_minutes=duration_minutes,
                purpose=purpose,
                status=AppointmentStatus.SCHEDULED,
            )

            created = await self.db.create_appointment(appointment)

            # Track in state
            self.state.appointments_booked.append({
                "id": created.id,
                "date": date,
                "time": time,
                "purpose": purpose,
            })

            # Format verbal confirmation
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                friendly_date = date_obj.strftime("%A, %B %d")
                time_obj = datetime.strptime(time, "%H:%M")
                friendly_time = time_obj.strftime("%I:%M %p").lstrip("0")
            except ValueError:
                friendly_date = date
                friendly_time = time

            purpose_text = f" for {purpose}" if purpose else ""
            verbal = f"I've booked your appointment for {friendly_date} at {friendly_time}{purpose_text}. Is there anything else I can help you with?"

            return ToolResult(
                success=True,
                data={"appointment": created.model_dump()},
                message=f"Appointment booked: {created.id}",
                verbal_response=verbal
            )

        except ValueError as e:
            logger.error(f"Booking validation error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response=str(e)
            )
        except Exception as e:
            logger.error(f"Error booking appointment: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble booking that appointment. Please try again."
            )

    async def retrieve_appointments(
        self,
        include_past: bool = False,
        status: Optional[str] = None,
    ) -> ToolResult:
        """
        Retrieve user's appointments.

        Args:
            include_past: Whether to include past appointments
            status: Filter by status

        Returns:
            ToolResult with appointments list
        """
        if not self.state:
            return ToolResult(
                success=False,
                error="Conversation not initialized",
                verbal_response="I'm sorry, there was a technical issue."
            )

        if not self.state.is_identified:
            return ToolResult(
                success=False,
                error="User not identified",
                verbal_response="I'll need your phone number to look up your appointments. What's your phone number?"
            )

        try:
            status_filter = None
            if status and status != "all":
                status_filter = AppointmentStatus(status)

            appointments = await self.db.get_appointments_by_phone(
                self.state.user_phone,
                status=status_filter,
                include_past=include_past,
            )

            if not appointments:
                return ToolResult(
                    success=True,
                    data={"appointments": []},
                    message="No appointments found",
                    verbal_response="You don't have any appointments scheduled. Would you like to book one?"
                )

            # Format verbal response
            upcoming = [a for a in appointments if a.status == AppointmentStatus.SCHEDULED]

            if len(upcoming) == 1:
                apt = upcoming[0]
                verbal = f"You have one appointment: {apt.to_verbal_summary()}."
            elif len(upcoming) > 1:
                verbal = f"You have {len(upcoming)} appointments. "
                for apt in upcoming[:3]:
                    verbal += f"{apt.to_verbal_summary()}. "
                if len(upcoming) > 3:
                    verbal += f"And {len(upcoming) - 3} more."
            else:
                verbal = "You don't have any upcoming appointments."

            return ToolResult(
                success=True,
                data={"appointments": [a.model_dump() for a in appointments]},
                message=f"Found {len(appointments)} appointments",
                verbal_response=verbal
            )

        except Exception as e:
            logger.error(f"Error retrieving appointments: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble retrieving your appointments. Please try again."
            )

    async def cancel_appointment(
        self,
        appointment_id: Optional[str] = None,
        date: Optional[str] = None,
        time: Optional[str] = None,
    ) -> ToolResult:
        """
        Cancel an appointment.

        Args:
            appointment_id: Specific appointment ID
            date: Date of appointment (if ID unknown)
            time: Time of appointment (if ID unknown)

        Returns:
            ToolResult with cancellation confirmation
        """
        if not self.state or not self.state.is_identified:
            return ToolResult(
                success=False,
                error="User not identified",
                verbal_response="I need to verify your phone number before I can cancel appointments."
            )

        try:
            # Find the appointment
            if appointment_id:
                apt = await self.db.get_appointment_by_id(appointment_id)
                if not apt or apt.user_phone != self.state.user_phone:
                    return ToolResult(
                        success=False,
                        error="Appointment not found",
                        verbal_response="I couldn't find that appointment. Could you tell me the date and time?"
                    )
            elif date and time:
                # Find by date/time
                appointments = await self.db.get_appointments_by_phone(
                    self.state.user_phone,
                    status=AppointmentStatus.SCHEDULED,
                )
                apt = next(
                    (a for a in appointments if a.date == date and a.time == time),
                    None
                )
                if not apt:
                    return ToolResult(
                        success=False,
                        error="Appointment not found",
                        verbal_response=f"I couldn't find an appointment on {date} at {time}. Would you like me to look up your appointments?"
                    )
            else:
                return ToolResult(
                    success=False,
                    error="No appointment specified",
                    verbal_response="Which appointment would you like to cancel? Please tell me the date and time."
                )

            # Cancel the appointment
            cancelled = await self.db.cancel_appointment(apt.id)

            # Track in state
            self.state.appointments_cancelled.append({
                "id": apt.id,
                "date": apt.date,
                "time": apt.time,
            })

            verbal = f"I've cancelled your appointment on {apt.date} at {apt.time}. Is there anything else I can help you with?"

            return ToolResult(
                success=True,
                data={"appointment": cancelled.model_dump()},
                message=f"Appointment {apt.id} cancelled",
                verbal_response=verbal
            )

        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble cancelling that appointment. Please try again."
            )

    async def modify_appointment(
        self,
        appointment_id: Optional[str] = None,
        current_date: Optional[str] = None,
        current_time: Optional[str] = None,
        new_date: Optional[str] = None,
        new_time: Optional[str] = None,
    ) -> ToolResult:
        """
        Modify an appointment's date or time.

        Args:
            appointment_id: Specific appointment ID
            current_date: Current date (if ID unknown)
            current_time: Current time (if ID unknown)
            new_date: New date
            new_time: New time

        Returns:
            ToolResult with modification confirmation
        """
        if not self.state or not self.state.is_identified:
            return ToolResult(
                success=False,
                error="User not identified",
                verbal_response="I need your phone number before I can modify appointments."
            )

        if not new_date and not new_time:
            return ToolResult(
                success=False,
                error="No changes specified",
                verbal_response="What would you like to change the appointment to? Please tell me the new date or time."
            )

        try:
            # Find the appointment
            if appointment_id:
                apt = await self.db.get_appointment_by_id(appointment_id)
                if not apt or apt.user_phone != self.state.user_phone:
                    return ToolResult(
                        success=False,
                        error="Appointment not found",
                        verbal_response="I couldn't find that appointment."
                    )
            elif current_date and current_time:
                appointments = await self.db.get_appointments_by_phone(
                    self.state.user_phone,
                    status=AppointmentStatus.SCHEDULED,
                )
                apt = next(
                    (a for a in appointments if a.date == current_date and a.time == current_time),
                    None
                )
                if not apt:
                    return ToolResult(
                        success=False,
                        error="Appointment not found",
                        verbal_response="I couldn't find that appointment. Would you like me to look up your appointments?"
                    )
            else:
                return ToolResult(
                    success=False,
                    error="No appointment specified",
                    verbal_response="Which appointment would you like to modify?"
                )

            # Validate new slot if changing
            target_date = new_date or apt.date
            target_time = new_time or apt.time

            is_valid, error_msg = self.slots.validate_slot(
                target_date, target_time, self.state.user_timezone
            )
            if not is_valid:
                return ToolResult(
                    success=False,
                    error=error_msg,
                    verbal_response=error_msg
                )

            # Modify the appointment
            modified = await self.db.modify_appointment(
                apt.id,
                new_date=new_date,
                new_time=new_time,
            )

            # Track in state
            self.state.appointments_modified.append({
                "id": apt.id,
                "old_date": apt.date,
                "old_time": apt.time,
                "new_date": target_date,
                "new_time": target_time,
            })

            verbal = f"I've updated your appointment to {target_date} at {target_time}. Is there anything else?"

            return ToolResult(
                success=True,
                data={"appointment": modified.model_dump()},
                message=f"Appointment {apt.id} modified",
                verbal_response=verbal
            )

        except ValueError as e:
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response=str(e)
            )
        except Exception as e:
            logger.error(f"Error modifying appointment: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble modifying that appointment. Please try again."
            )

    async def end_conversation(self, reason: str = "user_requested") -> ToolResult:
        """
        End the conversation and generate summary.

        Args:
            reason: Reason for ending

        Returns:
            ToolResult with summary data
        """
        if not self.state:
            return ToolResult(
                success=True,
                data={"reason": reason},
                message="Conversation ended",
                verbal_response="Goodbye! Have a great day."
            )

        self.state.should_end = True

        # Build summary data
        summary_data = {
            "reason": reason,
            "appointments_booked": self.state.appointments_booked,
            "appointments_modified": self.state.appointments_modified,
            "appointments_cancelled": self.state.appointments_cancelled,
            "user_identified": self.state.is_identified,
        }

        # Generate farewell
        if self.state.appointments_booked:
            apt = self.state.appointments_booked[-1]
            verbal = f"Great! Your appointment is confirmed for {apt['date']} at {apt['time']}. Have a wonderful day!"
        elif self.state.appointments_modified:
            apt = self.state.appointments_modified[-1]
            verbal = f"Your appointment has been updated to {apt['new_date']} at {apt['new_time']}. Take care!"
        elif self.state.appointments_cancelled:
            verbal = "Your appointment has been cancelled. Feel free to reach out when you'd like to schedule again. Goodbye!"
        else:
            verbal = "Thank you for calling. Have a great day!"

        return ToolResult(
            success=True,
            data=summary_data,
            message="Conversation ended",
            verbal_response=verbal
        )

    async def execute_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool parameters

        Returns:
            ToolResult from the tool execution
        """
        tool_map = {
            "identify_user": self.identify_user,
            "fetch_slots": self.fetch_slots,
            "book_appointment": self.book_appointment,
            "retrieve_appointments": self.retrieve_appointments,
            "cancel_appointment": self.cancel_appointment,
            "modify_appointment": self.modify_appointment,
            "end_conversation": self.end_conversation,
        }

        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
                verbal_response="I'm not sure how to do that."
            )

        try:
            return await tool_func(**tool_input)
        except TypeError as e:
            logger.error(f"Tool parameter error for {tool_name}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                verbal_response="I had trouble processing that request."
            )
