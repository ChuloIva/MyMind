# MyMind Therapeutic AI System - Deployment Guide

## Overview

MyMind is a comprehensive AI-powered therapeutic support system that processes audio sessions, extracts insights, and provides real-time therapeutic analysis. This guide will walk you through the complete setup and deployment process.

## System Architecture

The system consists of several interconnected modules:

- **Input Processing**: Audio transcription (Whisper) and speaker diarization (Pyannote)
- **Preprocessing**: GPT-4o keyword extraction and sentiment analysis  
- **Analysis Engine**: NLP processing, RAG system, and therapeutic methods evaluation
- **Profiling System**: Client assessment and trajectory tracking
- **Output Layer**: Report generation and streaming insights
- **API Gateway**: FastAPI-based REST API
- **Database**: PostgreSQL/SQLite with SQLModel
- **Frontend**: React-based dashboard with visualization

## Quick Start

### 1. Initial Setup

Run the automated setup script:

```bash
python startup.py
```

This script will:
- Check Python version compatibility
- Create required directories
- Install core dependencies
- Generate configuration files
- Create a minimal working application

### 2. Basic System Launch

Start the minimal system:

```bash
python minimal_app.py
```

Then visit: http://localhost:8000

You'll see a web interface with system status and API documentation.

## Full System Setup

### Prerequisites

- Python 3.8+
- Git
- At least 4GB RAM
- 10GB free disk space
- Internet connection for model downloads

### Required API Keys

1. **OpenAI API Key**
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Add to `.env` file: `OPENAI_API_KEY=your_key_here`

2. **HuggingFace Token**
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token
   - Accept user conditions for:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
   - Add to `.env` file: `HF_TOKEN=your_token_here`

### Full Dependency Installation

Install ML dependencies (requires additional system packages):

```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install python3-dev python3-venv ffmpeg

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Database Setup

#### Option 1: SQLite (Default - Easiest)
No additional setup required. The system will create a SQLite database automatically.

#### Option 2: PostgreSQL (Recommended for Production)

1. Install PostgreSQL:
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb mymind_db
sudo -u postgres createuser mymind_user
sudo -u postgres psql -c "ALTER USER mymind_user PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mymind_db TO mymind_user;"
```

2. Update `.env` file:
```
DATABASE_URL=postgresql://mymind_user:secure_password@localhost:5432/mymind_db
```

## Configuration

### Environment Variables

Edit the `.env` file to configure your system:

```bash
# Database Configuration
DATABASE_URL=sqlite:///./data/mymind.db

# API Keys (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here

# Model Configuration
WHISPER_MODEL=base          # Options: tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu          # Options: cpu, cuda
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1

# Processing Configuration  
CHUNK_SIZE=3                # Sentences to process together
BATCH_SIZE=4                # Processing batch size
MAX_SPEAKERS=5              # Maximum speakers to detect
```

### Hardware Recommendations

#### Minimum (CPU-only):
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB
- Model: `WHISPER_MODEL=base`, `WHISPER_DEVICE=cpu`

#### Recommended (GPU):
- CPU: 8 cores  
- RAM: 16GB
- GPU: 8GB VRAM (NVIDIA)
- Storage: 50GB
- Model: `WHISPER_MODEL=large-v3`, `WHISPER_DEVICE=cuda`

## Running the System

### Development Mode

```bash
# Start with auto-reload
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Start with production settings
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

1. Build Docker image:
```bash
docker build -t mymind-ai .
```

2. Run container:
```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data mymind-ai
```

## Usage Guide

### 1. Audio Upload and Processing

Upload therapy session audio:

```bash
curl -X POST "http://localhost:8000/api/preprocess/upload-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@session.wav" \
     -F "num_speakers=2"
```

### 2. Check Processing Status

```bash
curl "http://localhost:8000/api/preprocess/status/{session_id}"
```

### 3. Get Analysis Results

```bash
curl "http://localhost:8000/api/analysis/sessions/{session_id}/therapeutic-analysis"
```

### 4. Generate Reports

```bash
curl "http://localhost:8000/api/output/sessions/{session_id}/report?report_type=comprehensive"
```

## API Documentation

Once running, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

## Frontend Setup

### Dashboard Application

```bash
cd ui/dashboard
npm install
npm run dev
```

Access at: http://localhost:3000

### Chat Interface

```bash
cd ui/chat  
npm install
npm start
```

### Profile Management

```bash
cd ui/profile
npm install
npm run dev
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'package_name'
```
**Solution**: Install missing dependencies
```bash
pip install package_name
```

#### 2. CUDA/GPU Issues
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `BATCH_SIZE` in `.env`
- Use smaller model: `WHISPER_MODEL=base`
- Switch to CPU: `WHISPER_DEVICE=cpu`

#### 3. Pyannote Authentication
```
ValueError: HuggingFace token is required
```
**Solution**: 
- Get token from https://huggingface.co/settings/tokens
- Accept model user agreements
- Add to `.env`: `HF_TOKEN=your_token_here`

#### 4. Database Connection Issues
```
sqlalchemy.exc.OperationalError
```
**Solutions**:
- Check database server is running
- Verify connection string in `.env`
- For PostgreSQL: ensure user has permissions

#### 5. Audio Processing Fails
```
FileNotFoundError: ffmpeg not found
```
**Solution**: Install ffmpeg
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Performance Optimization

#### 1. Speed Up Processing
- Use GPU: `WHISPER_DEVICE=cuda`
- Increase batch size: `BATCH_SIZE=24`
- Use faster models: `WHISPER_MODEL=base`

#### 2. Reduce Memory Usage
- Decrease batch size: `BATCH_SIZE=4`
- Use smaller models: `WHISPER_MODEL=tiny`
- Process shorter audio segments

#### 3. Improve Accuracy
- Use larger models: `WHISPER_MODEL=large-v3`
- Provide speaker count: `num_speakers=2`
- Use higher quality audio (16kHz+)

## Security Considerations

### API Security
- Change default `SECRET_KEY` in `.env`
- Use HTTPS in production
- Implement rate limiting
- Add authentication/authorization

### Data Privacy
- Audio files are processed locally
- No data sent to external services (except OpenAI/HF APIs)
- Implement data retention policies
- Encrypt sensitive data

### Production Deployment
- Use environment-specific configuration
- Set up monitoring and logging
- Implement backup strategies
- Use reverse proxy (nginx)

## Monitoring and Logging

### Application Logs
```bash
# View logs
tail -f logs/application.log

# Set log level in code
logging.basicConfig(level=logging.INFO)
```

### System Monitoring
- Monitor CPU/GPU usage
- Track memory consumption
- Monitor disk space
- Set up health checks

## Support and Documentation

### Getting Help
- Check API documentation: `/docs`
- Review error logs: `logs/`
- Consult README files in each module
- Check GitHub issues/discussions

### Additional Resources
- OpenAI API Documentation: https://platform.openai.com/docs
- Pyannote Documentation: https://pyannote.github.io/pyannote-audio/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- React Documentation: https://reactjs.org/docs/

## System Architecture Details

### Processing Pipeline
1. **Audio Upload** → Save to `data/raw_audio/`
2. **Transcription** → Whisper processing with timestamps
3. **Speaker Diarization** → Pyannote speaker identification
4. **Keyword Extraction** → GPT-4o analysis of segments
5. **Therapeutic Analysis** → CBT/Schema therapy evaluation
6. **Visualization** → UMAP/t-SNE concept mapping
7. **Report Generation** → Structured therapeutic insights

### Data Flow
```
Audio File → Transcription → Segmentation → Analysis → Storage → API → Frontend
```

### Module Dependencies
```
Frontend ←→ API Gateway ←→ Analysis Engine ←→ Database
                ↓              ↓
        Background Tasks ←→ ML Models
```

This comprehensive system provides a complete therapeutic AI platform with professional-grade analysis capabilities while maintaining flexibility for different deployment scenarios.