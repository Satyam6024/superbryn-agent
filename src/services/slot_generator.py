"""Slot generator for available appointment times."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from ..models import TimeSlot

logger = logging.getLogger(__name__)


class SlotGenerator:
    """Generates available appointment slots based on configuration."""

    def __init__(
        self,
        business_hours_start: int = 8,
        business_hours_end: int = 20,
        business_days: List[int] = None,
        booking_advance_days: int = 30,
        default_slot_duration: int = 30,
    ):
        """
        Initialize slot generator.

        Args:
            business_hours_start: Start hour (24-hour format), default 8 AM
            business_hours_end: End hour (24-hour format), default 8 PM
            business_days: List of weekday numbers (0=Monday), default Mon-Sat
            booking_advance_days: How many days ahead to allow booking
            default_slot_duration: Default slot duration in minutes
        """
        self.business_hours_start = business_hours_start
        self.business_hours_end = business_hours_end
        self.business_days = business_days or [0, 1, 2, 3, 4, 5]  # Mon-Sat
        self.booking_advance_days = booking_advance_days
        self.default_slot_duration = default_slot_duration

    def generate_slots(
        self,
        user_timezone: str = "UTC",
        duration_minutes: Optional[int] = None,
        booked_slots: Optional[List[tuple]] = None,
    ) -> List[TimeSlot]:
        """
        Generate all available slots for the booking period.

        Args:
            user_timezone: User's timezone string (e.g., "America/New_York")
            duration_minutes: Requested slot duration
            booked_slots: List of (date, time) tuples that are already booked

        Returns:
            List of available TimeSlot objects
        """
        duration = duration_minutes or self.default_slot_duration
        booked_set = set(booked_slots or [])

        try:
            tz = ZoneInfo(user_timezone)
        except Exception:
            logger.warning(f"Invalid timezone {user_timezone}, using UTC")
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        slots = []

        for day_offset in range(self.booking_advance_days):
            current_date = now.date() + timedelta(days=day_offset)

            # Skip non-business days
            if current_date.weekday() not in self.business_days:
                continue

            # Generate time slots for this day
            for hour in range(self.business_hours_start, self.business_hours_end):
                for minute in [0, 30]:  # 30-minute intervals
                    # Skip if this slot wouldn't fit within business hours
                    slot_end_hour = hour + (minute + duration) // 60
                    slot_end_minute = (minute + duration) % 60
                    if slot_end_hour > self.business_hours_end or (
                        slot_end_hour == self.business_hours_end and slot_end_minute > 0
                    ):
                        continue

                    date_str = current_date.strftime("%Y-%m-%d")
                    time_str = f"{hour:02d}:{minute:02d}"

                    # Skip if in the past (for today)
                    if day_offset == 0:
                        slot_datetime = datetime.combine(
                            current_date,
                            datetime.strptime(time_str, "%H:%M").time(),
                            tzinfo=tz
                        )
                        if slot_datetime <= now + timedelta(hours=1):  # 1 hour buffer
                            continue

                    # Check if slot is booked
                    is_available = (date_str, time_str) not in booked_set

                    slots.append(TimeSlot(
                        date=date_str,
                        time=time_str,
                        duration_minutes=duration,
                        is_available=is_available,
                    ))

        return slots

    def get_available_slots(
        self,
        user_timezone: str = "UTC",
        duration_minutes: Optional[int] = None,
        booked_slots: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
    ) -> List[TimeSlot]:
        """
        Get only available (not booked) slots.

        Args:
            user_timezone: User's timezone
            duration_minutes: Requested slot duration
            booked_slots: Already booked (date, time) tuples
            limit: Maximum number of slots to return

        Returns:
            List of available TimeSlot objects
        """
        all_slots = self.generate_slots(user_timezone, duration_minutes, booked_slots)
        available = [s for s in all_slots if s.is_available]

        if limit:
            return available[:limit]
        return available

    def get_slots_for_date(
        self,
        date: str,
        user_timezone: str = "UTC",
        booked_slots: Optional[List[tuple]] = None,
    ) -> List[TimeSlot]:
        """Get available slots for a specific date."""
        all_slots = self.get_available_slots(user_timezone, booked_slots=booked_slots)
        return [s for s in all_slots if s.date == date]

    def format_slots_for_speech(
        self,
        slots: List[TimeSlot],
        max_slots: int = 5,
    ) -> str:
        """
        Format slots for verbal output.

        Args:
            slots: List of TimeSlot objects
            max_slots: Maximum slots to include in speech

        Returns:
            Human-readable string for TTS
        """
        if not slots:
            return "I don't have any available slots at the moment."

        display_slots = slots[:max_slots]
        total = len(slots)

        # Group by date for cleaner output
        by_date = {}
        for slot in display_slots:
            if slot.date not in by_date:
                by_date[slot.date] = []
            by_date[slot.date].append(slot.time)

        parts = []
        for date_str, times in by_date.items():
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                friendly_date = date_obj.strftime("%A, %B %d")
            except ValueError:
                friendly_date = date_str

            # Format times
            formatted_times = []
            for t in times:
                try:
                    time_obj = datetime.strptime(t, "%H:%M")
                    formatted_times.append(time_obj.strftime("%I:%M %p").lstrip("0"))
                except ValueError:
                    formatted_times.append(t)

            if len(formatted_times) == 1:
                parts.append(f"{friendly_date} at {formatted_times[0]}")
            else:
                times_str = ", ".join(formatted_times[:-1]) + f" or {formatted_times[-1]}"
                parts.append(f"{friendly_date} at {times_str}")

        result = "; ".join(parts)

        if total > max_slots:
            result += f". I have {total - max_slots} more slots available if these don't work for you."

        return result

    def validate_slot(
        self,
        date: str,
        time: str,
        user_timezone: str = "UTC",
    ) -> tuple[bool, str]:
        """
        Validate if a date/time is a valid bookable slot.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return False, "Invalid date format. Please use YYYY-MM-DD."

        try:
            time_obj = datetime.strptime(time, "%H:%M").time()
        except ValueError:
            return False, "Invalid time format. Please use HH:MM."

        # Check if within business hours
        if time_obj.hour < self.business_hours_start or time_obj.hour >= self.business_hours_end:
            return False, f"That time is outside business hours ({self.business_hours_start}:00 - {self.business_hours_end}:00)."

        # Check if business day
        if date_obj.weekday() not in self.business_days:
            return False, "That date is not a business day. We're open Monday through Saturday."

        # Check if not in the past
        try:
            tz = ZoneInfo(user_timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        slot_datetime = datetime.combine(date_obj, time_obj, tzinfo=tz)

        if slot_datetime <= now:
            return False, "That time is in the past. Please choose a future time."

        # Check if within booking window
        max_date = now.date() + timedelta(days=self.booking_advance_days)
        if date_obj > max_date:
            return False, f"That's too far in advance. You can book up to {self.booking_advance_days} days ahead."

        return True, ""
