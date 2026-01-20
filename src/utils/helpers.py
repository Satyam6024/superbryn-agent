"""Helper utility functions."""

import re
from datetime import datetime
from typing import Optional, Tuple
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


def format_datetime(dt: datetime, format_type: str = "friendly") -> str:
    """
    Format datetime for display.

    Args:
        dt: Datetime object
        format_type: "friendly", "iso", or "short"

    Returns:
        Formatted datetime string
    """
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "short":
        return dt.strftime("%m/%d %I:%M %p")
    else:  # friendly
        return dt.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")


def parse_user_datetime(text: str, reference: Optional[datetime] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse natural language date/time into structured format.

    Args:
        text: User's input text (e.g., "tomorrow at 3pm", "next Monday")
        reference: Reference datetime for relative dates

    Returns:
        Tuple of (date_str in YYYY-MM-DD, time_str in HH:MM) or (None, None)
    """
    if reference is None:
        reference = datetime.now()

    text = text.lower().strip()

    try:
        # Handle relative dates
        if "tomorrow" in text:
            date = reference + relativedelta(days=1)
        elif "today" in text:
            date = reference
        elif "next week" in text:
            date = reference + relativedelta(weeks=1)
        else:
            # Try to parse with dateutil
            date = date_parser.parse(text, fuzzy=True, default=reference)

        date_str = date.strftime("%Y-%m-%d")

        # Extract time if present
        time_str = None
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)

            if period:
                if period.lower() == 'pm' and hour < 12:
                    hour += 12
                elif period.lower() == 'am' and hour == 12:
                    hour = 0

            time_str = f"{hour:02d}:{minute:02d}"

        return date_str, time_str

    except (ValueError, TypeError):
        return None, None


def sanitize_phone(phone: str) -> str:
    """
    Sanitize and format phone number.

    Args:
        phone: Raw phone number input

    Returns:
        Normalized phone number
    """
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


def format_phone_for_display(phone: str) -> str:
    """
    Format phone number for display.

    Args:
        phone: Normalized phone number

    Returns:
        Human-readable format
    """
    digits = re.sub(r'\D', '', phone)

    if len(digits) == 11 and digits.startswith('1'):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    elif len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

    return phone


def calculate_duration_seconds(start: datetime, end: datetime) -> int:
    """Calculate duration in seconds between two datetimes."""
    delta = end - start
    return int(delta.total_seconds())


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
