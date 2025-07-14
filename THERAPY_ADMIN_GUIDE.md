# MyMind Therapy Admin System

A comprehensive therapy practice management system with an intuitive UI/UX for managing clients, sessions, and therapeutic insights.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation & Setup

1. **Clone and Navigate**
   ```bash
   cd /home/engine/project
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn sqlmodel pydantic-settings requests python-dotenv
   ```

4. **Initialize Database**
   ```bash
   python init_db.py
   ```

5. **Start Server**
   ```bash
   python run_therapy_admin.py
   ```

6. **Access Application**
   - **Main Admin UI**: http://localhost:8000
   - **Analysis UI**: http://localhost:8000/analysis
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

## 📋 System Architecture

### UI/UX Structure
```
HOME PAGE
│
├─ CLIENTS VIEW
│  │
│  ├─ Clients (List)
│  │  │
│  │  └─ New Client
│  │     │
│  │     ├─ Profile → Stats for this client → Fill in questionnaire
│  │     │
│  │     └─ Sessions
│  │        │
│  │        ├─ Overview of sessions (snapshots)
│  │        │
│  │        └─ New Session
│  │           │
│  │           └─ Upload | record | ──────────────┐
│  │                                              │
│  │                                              ▼
│  │                                         SESSION VIEW
│  │                                              │
│  │                                              └─ Chat
│  │
├─ CALENDAR
│  │
│  └─ Book / organise sessions / clients
│
└─ SETTINGS
```

### Database Models

#### Client
- `id`: UUID (Primary Key)
- `name`: String
- `email`: String (Optional)
- `created_at`: DateTime
- `updated_at`: DateTime

#### Session
- `id`: UUID (Primary Key)
- `client_id`: UUID (Foreign Key to Client)
- `title`: String (Optional)
- `status`: String (pending, processing, completed, failed)
- `notes`: String (Optional)
- `created_at`: DateTime
- `updated_at`: DateTime

#### SessionSentence
- `id`: UUID (Primary Key)
- `session_id`: UUID (Foreign Key to Session)
- `sentence_index`: Integer
- `start_ms`: Integer
- `end_ms`: Integer
- `speaker`: String
- `text`: String
- `confidence`: Float (Optional)

## 🎯 Key Features

### 1. Dashboard (Home)
- **Statistics Overview**: Total clients, sessions, daily activity
- **Recent Activity**: Latest clients and sessions
- **Quick Actions**: Create new client, view all clients, schedule sessions

### 2. Client Management
- **Client List**: View all clients with session counts
- **Client Profile**: Detailed client information
- **Client Stats**: Visual analytics and progress tracking
- **Assessment Questionnaire**: Standardized client assessment forms

### 3. Session Management
- **Session Creation**: Create new therapy sessions
- **Session Overview**: View session details and status
- **File Upload**: Upload audio files for analysis
- **Session Chat**: AI-powered chat for session insights

### 4. Calendar
- **Session Scheduling**: Book and organize appointments
- **Calendar View**: Monthly calendar with session events
- **Date Selection**: Interactive date picker

### 5. Settings
- **Practice Configuration**: Clinic name, therapist details
- **Session Defaults**: Default session duration, timezone
- **System Preferences**: Application settings

## 🔧 API Endpoints

### Client Management
- `GET /api/clients/` - List all clients
- `POST /api/clients/` - Create new client
- `GET /api/clients/{id}` - Get specific client
- `PUT /api/clients/{id}` - Update client
- `DELETE /api/clients/{id}` - Delete client

### Session Management
- `GET /api/sessions/` - List all sessions
- `POST /api/clients/{id}/sessions` - Create session for client
- `GET /api/sessions/{id}` - Get specific session
- `PUT /api/sessions/{id}` - Update session
- `DELETE /api/sessions/{id}` - Delete session

### File Processing
- `POST /api/preprocess/upload-audio` - Upload audio file
- `GET /api/preprocess/status/{session_id}` - Check processing status

## 💡 Usage Examples

### Creating a New Client
1. Navigate to **Clients** → **New Client**
2. Fill in client name and email
3. Click **Create Client**
4. Client profile page opens automatically

### Creating a Session
1. From client profile, click **Sessions**
2. Click **New Session**
3. Enter session title and notes
4. Upload audio file or start recording
5. Session view opens with chat interface

### Uploading Audio Files
1. In session creation, drag and drop audio file
2. Supported formats: WAV, MP3, M4A, FLAC, OGG
3. File uploads automatically to session
4. Processing status updates in real-time

## 🎨 UI Components

### Navigation
- **Sidebar**: Main navigation with Home, Clients, Calendar, Settings
- **Breadcrumbs**: Shows current page hierarchy
- **Quick Actions**: Context-sensitive action buttons

### Forms
- **Client Creation**: Name, email fields with validation
- **Session Creation**: Title, notes, file upload
- **Questionnaire**: Multi-section assessment form

### Data Display
- **Statistics Cards**: Key metrics with gradient backgrounds
- **Client Cards**: Clickable client information panels
- **Session List**: Chronological session overview
- **Calendar Grid**: Interactive monthly calendar

### Interactive Elements
- **Drag & Drop**: File upload with visual feedback
- **Chat Interface**: Real-time AI conversation
- **Modal Windows**: Loading overlays and confirmations
- **Toast Messages**: Success/error notifications

## 🔒 Security Features

### Data Protection
- UUID-based identifiers for all records
- Input validation and sanitization
- Secure file upload handling
- Database connection pooling

### Error Handling
- Comprehensive error messages
- Graceful degradation
- Database connection fallbacks
- Client-side validation

## 📊 Analytics & Insights

### Client Statistics
- Session frequency and duration
- Progress tracking over time
- Mood and assessment trends
- Visual charts and graphs

### Session Analytics
- Audio processing status
- Transcription quality metrics
- Sentiment analysis results
- Therapeutic pattern recognition

## 🛠️ Development

### File Structure
```
/home/engine/project/
├── src/
│   ├── api/
│   │   ├── main_simple.py          # Simplified API server
│   │   └── routers/
│   │       ├── client_management.py
│   │       └── session_management.py
│   ├── database/
│   │   ├── database.py
│   │   └── models.py
│   └── common/
│       └── config.py
├── therapy_admin.html              # Main UI
├── therapy_admin.js               # Frontend logic
├── run_therapy_admin.py           # Server launcher
├── init_db.py                     # Database initialization
└── test_db_simple.py              # Database tests
```

### Testing
```bash
# Test database operations
python test_db_simple.py

# Test API endpoints (when server is running)
python test_api.py
```

### Development Server
```bash
# Start with auto-reload
python run_therapy_admin.py

# Manual server start
python -m uvicorn src.api.main_simple:app --reload --host 0.0.0.0 --port 8000
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Run `python init_db.py` to initialize database
   - Check if `mymind.db` exists in project root

2. **Server Won't Start**
   - Ensure virtual environment is activated
   - Install missing dependencies: `pip install -r requirements.txt`

3. **UI Not Loading**
   - Check if server is running on port 8000
   - Verify `therapy_admin.html` and `therapy_admin.js` exist

4. **API Endpoints Not Working**
   - Visit http://localhost:8000/docs for API documentation
   - Check server logs for error messages

### Debug Mode
```bash
# Run with debug logging
python run_therapy_admin.py --log-level debug
```

## 📚 Future Enhancements

### Planned Features
- **Real-time Audio Recording**: Browser-based audio capture
- **Advanced Analytics**: ML-powered insights and recommendations
- **Multi-user Support**: Role-based access control
- **Integration APIs**: Connect with external therapy tools
- **Mobile App**: React Native companion app
- **Backup & Sync**: Cloud storage integration

### Extensibility
- **Plugin System**: Custom analysis modules
- **Theme Support**: Customizable UI themes
- **Webhook Support**: Real-time notifications
- **Export Functions**: PDF reports and data export

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please:
1. Check this documentation
2. Review the troubleshooting section
3. Check server logs for error messages
4. Create an issue with detailed error information

---

**MyMind Therapy Admin System** - Empowering therapeutic practice with intelligent technology. 🧠✨