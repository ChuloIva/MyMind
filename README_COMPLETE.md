# 🧠 MyMind - Therapeutic AI Platform

A comprehensive therapeutic AI platform that analyzes therapy sessions using multiple AI techniques including keyword extraction, cognitive bias detection, schema analysis, and needs-based profiling.

## ✨ Features

### 📝 **Text Analysis** (Ready to Use)
- **Keywords & Sentiment Analysis**: Extract key themes and emotional patterns
- **Cognitive Distortion Detection (CBT)**: Identify cognitive biases and distortions
- **Schema Pattern Analysis**: Detect underlying psychological schemas
- **Needs Analysis**: Map content to universal psychological needs (requires OpenAI API key)

### 🎵 **Audio Session Processing** (Basic Implementation)
- Audio file upload (.wav, .mp3, .m4a, .flac, .ogg)
- Session management and tracking
- Status monitoring

### 🎯 **Needs-Based Profiling** (Advanced Feature)
- Comprehensive client profiling based on psychological needs
- Visual dashboards with radar and bar charts
- Therapeutic insights and recommendations
- Life segment analysis across different areas

### 📊 **Session Management**
- Track processing status
- Client-based session organization
- Progress monitoring

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd /Users/ivanculo/Desktop/Projects/MyMind

# Initialize database
python init_db.py

# Load reference data for needs profiling
python scripts/load_reference_data.py
```

### 2. Start the Server
```bash
# Start the comprehensive UI server
python run_server.py
```

### 3. Access the Platform
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

## 📱 Using the Platform

### **Text Analysis Tab**
1. Upload a `.txt` file containing therapy session transcript
2. Click "🔍 Analyze Text"
3. View results for:
   - Keywords & Sentiment
   - Cognitive Distortions (CBT)
   - Schema Patterns
   - Needs Analysis (if OpenAI API configured)

**Sample file**: Use `sample_session.txt` for testing

### **Audio Sessions Tab**
1. Upload audio file (.wav, .mp3, .m4a, .flac, .ogg)
2. Optionally specify client ID and number of speakers
3. Click "⬆️ Upload Audio" then "🚀 Complete Processing"
4. Monitor progress in Session Management tab

### **Needs Profiling Tab**
1. Enter a Client ID
2. Click "🎯 Generate Profile" to analyze client's sessions
3. Click "📊 View Dashboard" to see visual analytics including:
   - Life Segments Radar Chart
   - Needs Fulfillment Bar Chart
   - Therapeutic Insights
   - Recommendations

## 🔧 API Endpoints

### Core Analysis
- `POST /api/simple_analyze` - Comprehensive text analysis
- `GET /api/docs` - API documentation

### Audio Processing
- `POST /api/preprocess/upload-audio` - Upload audio file
- `GET /api/preprocess/status/{session_id}` - Check processing status

### Needs Profiling
- `POST /api/profiling/clients/{client_id}/analyze-needs` - Generate needs profile
- `GET /api/profiling/clients/{client_id}/needs-dashboard` - Get dashboard data

## 🎯 Working Features Status

### ✅ **Fully Implemented & Tested**
- Text file upload and analysis
- Keywords extraction with sentiment
- Cognitive distortion detection (CBT)
- Schema pattern analysis
- Database models and initialization
- Reference data loading (64 needs + 36 life segments)
- Basic profiling API endpoints
- Comprehensive responsive UI

### ⚠️ **Partially Implemented**
- Audio transcription (endpoints exist, may need speech-to-text service)
- Complete audio processing pipeline
- Session management UI

### 🔑 **Requires Configuration**
- Needs extraction (requires OpenAI API key in settings)
- Audio transcription (may require additional services)

## 📊 Database Schema

The platform uses SQLite with the following key tables:
- `client` - Client information
- `session` - Therapy sessions
- `needcategory` - Universal and SDT psychological needs (64 categories)
- `lifesegment` - Life areas for analysis (36 segments)
- `clientneedprofile` - Individual need extractions
- `clientneedsummary` - Aggregated client profiles

## 🧪 Testing the Platform

### Basic Text Analysis Test
1. Start server: `python run_server.py`
2. Open http://localhost:8000
3. Upload `sample_session.txt` in Text Analysis tab
4. Verify you get results for keywords, CBT, and schemas

### Database Verification
```bash
python -c "
from src.database.database import get_session
from src.database.models import NeedCategory, LifeSegment
from sqlmodel import select

db = next(get_session())
needs_count = len(db.exec(select(NeedCategory)).all())
segments_count = len(db.exec(select(LifeSegment)).all())
print(f'Needs: {needs_count}, Segments: {segments_count}')
"
```

## 🛠 Troubleshooting

### Server Won't Start
- Check Python path: `python run_server.py`
- Verify database: `python init_db.py`
- Check dependencies in requirements.txt

### Analysis Not Working
- Ensure sample file is properly formatted text
- Check browser console for JavaScript errors
- Verify API endpoints at http://localhost:8000/docs

### Needs Profiling Issues
- Requires OpenAI API key for needs extraction
- Generate profile before viewing dashboard
- Ensure client has session data

## 📁 Project Structure

```
MyMind/
├── src/
│   ├── api/                    # FastAPI application
│   ├── database/              # Database models and connections
│   ├── preprocessing/         # Text processing and extraction
│   ├── analysis/             # CBT and schema analysis
│   ├── profiling/            # Needs-based profiling
│   └── output/               # Report generation
├── scripts/                  # Utility scripts
├── ui/                      # Frontend components (partial)
├── index.html              # Main web interface
├── app.js                  # Frontend JavaScript
├── run_server.py          # Server launcher
└── sample_session.txt     # Test data
```

## 🎉 Next Steps

The platform is ready for testing and development. Key areas for enhancement:
1. Complete audio transcription integration
2. Enhanced session management features
3. Advanced visualization components
4. User authentication and multi-tenancy
5. Export capabilities for reports

## 🚨 Important Notes

- The platform currently runs on localhost:8000
- Database is SQLite-based for development
- Needs extraction requires OpenAI API key
- Audio features are basic implementation
- UI is responsive and modern

**Ready to use for therapy session text analysis and needs profiling!** 🎯