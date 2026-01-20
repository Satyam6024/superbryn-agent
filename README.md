# SuperBryn Voice Agent Backend

AI-powered voice agent for appointment booking using LiveKit, Deepgram, Cartesia, and Claude.

## Features

- **Real-time Voice Conversation**: Natural speech-to-speech interaction
- **Appointment Management**: Book, retrieve, modify, and cancel appointments
- **Smart Context**: Maintains conversation context across turns
- **Tool Calling**: Claude-powered intelligent tool selection
- **Database Integration**: Supabase for persistent storage

## Tech Stack

- **Voice Framework**: LiveKit Agents (Python)
- **Speech-to-Text**: Deepgram Nova-2
- **Text-to-Speech**: Cartesia
- **LLM**: Claude (Anthropic)
- **Database**: Supabase
- **API Server**: aiohttp

## Project Structure

```
superbryn-agent/
├── config/
│   └── settings.py       # Configuration management
├── src/
│   ├── agents/
│   │   └── voice_agent.py    # Main agent implementation
│   ├── api/
│   │   └── routes.py         # HTTP API endpoints
│   ├── models/
│   │   ├── appointment.py    # Appointment data models
│   │   ├── user.py           # User data models
│   │   └── conversation.py   # Conversation/logging models
│   ├── services/
│   │   ├── supabase_service.py   # Database operations
│   │   ├── claude_service.py     # LLM interactions
│   │   └── slot_generator.py     # Appointment slot generation
│   ├── tools/
│   │   └── appointment_tools.py  # Tool implementations
│   ├── utils/
│   │   └── helpers.py        # Utility functions
│   └── main.py               # Entry point
├── tests/
├── .env.example
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/superbryn-agent.git
cd superbryn-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Set up Supabase tables

Run the SQL migrations in your Supabase project (see Database Schema below).

### 6. Run the agent

```bash
# Run agent worker only
python -m src.main --mode agent

# Run API server only
python -m src.main --mode api

# Run both
python -m src.main --mode both
```

## Database Schema

Create these tables in your Supabase project:

```sql
-- Users table
CREATE TABLE users (
    phone_number TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_interaction TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}',
    total_appointments INTEGER DEFAULT 0
);

-- Appointments table
CREATE TABLE appointments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_phone TEXT REFERENCES users(phone_number),
    user_name TEXT,
    date DATE NOT NULL,
    time TIME NOT NULL,
    duration_minutes INTEGER DEFAULT 30,
    purpose TEXT,
    status TEXT DEFAULT 'scheduled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Tool call logs
CREATE TABLE tool_call_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters JSONB,
    result JSONB,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    duration_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation summaries
CREATE TABLE conversation_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_phone TEXT,
    user_name TEXT,
    summary_text TEXT,
    key_points JSONB DEFAULT '[]',
    appointments_booked JSONB DEFAULT '[]',
    appointments_modified JSONB DEFAULT '[]',
    appointments_cancelled JSONB DEFAULT '[]',
    user_preferences JSONB DEFAULT '{}',
    total_turns INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    duration_seconds INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Event logs
CREATE TABLE event_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB DEFAULT '{}',
    severity TEXT DEFAULT 'info',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_appointments_user_phone ON appointments(user_phone);
CREATE INDEX idx_appointments_date ON appointments(date);
CREATE INDEX idx_appointments_status ON appointments(status);
CREATE INDEX idx_tool_logs_session ON tool_call_logs(session_id);
CREATE INDEX idx_summaries_user ON conversation_summaries(user_phone);
```

## API Endpoints

### Health Check
```
GET /health
```

### Generate LiveKit Token
```
POST /api/token
Body: { "room_name": "optional", "participant_name": "optional", "user_timezone": "UTC" }
```

### Get Conversation History
```
GET /api/history/{phone}?limit=10
```

### Admin Endpoints (require X-Admin-Password header)
```
POST /api/admin/auth
Body: { "password": "admin-password" }

GET /api/admin/appointments?limit=100&offset=0
GET /api/admin/stats
```

## Deployment

### Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the `render.yaml` blueprint or configure manually
4. Add all required environment variables
5. Deploy

### Docker

```bash
docker build -t superbryn-agent .
docker run -p 8080:8080 --env-file .env superbryn-agent
```

## License

MIT
