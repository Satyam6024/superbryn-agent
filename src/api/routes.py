"""
API routes for the voice agent backend.
Provides endpoints for:
- Health checks
- Admin panel data
- Conversation history
- LiveKit token generation
"""

import logging
import os
from datetime import datetime
from typing import Optional

from aiohttp import web
from livekit.api import AccessToken, VideoGrants

logger = logging.getLogger(__name__)


def create_app(
    supabase_service,
    livekit_api_key: str,
    livekit_api_secret: str,
    admin_password: str,
) -> web.Application:
    """
    Create the aiohttp application with routes.

    Args:
        supabase_service: SupabaseService instance
        livekit_api_key: LiveKit API key
        livekit_api_secret: LiveKit API secret
        admin_password: Password for admin panel

    Returns:
        Configured aiohttp Application
    """
    app = web.Application()

    # Store services in app
    app["supabase"] = supabase_service
    app["livekit_api_key"] = livekit_api_key
    app["livekit_api_secret"] = livekit_api_secret
    app["admin_password"] = admin_password

    # Add routes
    app.router.add_get("/health", health_check)
    app.router.add_post("/api/token", generate_token)
    app.router.add_get("/api/history/{phone}", get_conversation_history)
    app.router.add_get("/api/admin/appointments", get_all_appointments)
    app.router.add_get("/api/admin/stats", get_admin_stats)
    app.router.add_post("/api/admin/auth", admin_auth)

    # CORS middleware
    app.middlewares.append(cors_middleware)

    return app


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Handle CORS for frontend requests."""
    # Handle preflight
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as e:
            response = e

    # Add CORS headers
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Admin-Password"
    response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "superbryn-agent",
    })


async def generate_token(request: web.Request) -> web.Response:
    """
    Generate a LiveKit access token for a client.

    Request body:
    {
        "room_name": "optional-room-name",
        "participant_name": "optional-name",
        "user_timezone": "America/New_York"
    }
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    room_name = data.get("room_name") or f"bryn-room-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    participant_name = data.get("participant_name") or f"user-{datetime.utcnow().timestamp()}"
    user_timezone = data.get("user_timezone", "UTC")

    api_key = request.app["livekit_api_key"]
    api_secret = request.app["livekit_api_secret"]

    # Create access token
    token = AccessToken(api_key, api_secret)
    token.identity = participant_name
    token.name = participant_name

    # Grant permissions
    grant = VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    token.video_grants = grant

    jwt = token.to_jwt()

    return web.json_response({
        "token": jwt,
        "room_name": room_name,
        "participant_name": participant_name,
        "user_timezone": user_timezone,
    })


async def get_conversation_history(request: web.Request) -> web.Response:
    """
    Get conversation history for a user.

    URL params:
    - phone: User's phone number

    Query params:
    - limit: Max number of summaries (default 10)
    """
    phone = request.match_info.get("phone")
    if not phone:
        return web.json_response({"error": "Phone number required"}, status=400)

    limit = int(request.query.get("limit", 10))
    db = request.app["supabase"]

    try:
        summaries = await db.get_conversation_history(phone, limit=limit)
        return web.json_response({
            "phone": phone,
            "conversations": [s.to_display_dict() for s in summaries],
        })
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return web.json_response({"error": "Failed to fetch history"}, status=500)


async def admin_auth(request: web.Request) -> web.Response:
    """
    Authenticate admin user.

    Request body:
    {
        "password": "admin-password"
    }
    """
    try:
        data = await request.json()
        password = data.get("password")

        if password == request.app["admin_password"]:
            return web.json_response({"authenticated": True})
        else:
            return web.json_response({"authenticated": False}, status=401)

    except Exception as e:
        logger.error(f"Admin auth error: {e}")
        return web.json_response({"error": "Authentication failed"}, status=500)


def _check_admin_auth(request: web.Request) -> bool:
    """Check if request has valid admin authentication."""
    password = request.headers.get("X-Admin-Password")
    return password == request.app["admin_password"]


async def get_all_appointments(request: web.Request) -> web.Response:
    """
    Get all appointments (admin only).

    Query params:
    - limit: Max appointments (default 100)
    - offset: Pagination offset (default 0)
    """
    if not _check_admin_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    limit = int(request.query.get("limit", 100))
    offset = int(request.query.get("offset", 0))
    db = request.app["supabase"]

    try:
        appointments = await db.get_all_appointments(limit=limit, offset=offset)
        total = await db.get_appointments_count()

        return web.json_response({
            "appointments": [apt.model_dump() for apt in appointments],
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Error fetching appointments: {e}")
        return web.json_response({"error": "Failed to fetch appointments"}, status=500)


async def get_admin_stats(request: web.Request) -> web.Response:
    """
    Get admin dashboard statistics.
    """
    if not _check_admin_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    db = request.app["supabase"]

    try:
        # Get various stats
        total_appointments = await db.get_appointments_count()

        # Get appointments by status
        scheduled = len(await db.get_all_appointments(limit=1000))  # Simplified

        return web.json_response({
            "total_appointments": total_appointments,
            "stats": {
                "total": total_appointments,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return web.json_response({"error": "Failed to fetch stats"}, status=500)
