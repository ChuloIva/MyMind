#!/usr/bin/env python3
"""
MyMind Therapeutic AI System Startup Script

This script sets up and runs the complete therapeutic AI system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_dependencies(from_file=False):
    """Install required Python packages"""
    logger.info("Installing dependencies...")

    if from_file:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install dependencies from requirements.txt: {e}")
    else:
        core_packages = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "python-multipart==0.0.6",
            "pydantic==2.4.2",
            "pydantic-settings==2.0.3",
            "python-dotenv==1.0.0",
            "requests==2.31.0",
            "httpx==0.25.2",
            "typing-extensions==4.8.0"
        ]
        for package in core_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")
    
    logger.info("Core dependencies installed")

def setup_directories():
    """Create required directories"""
    directories = [
        "./data",
        "./data/raw_audio",
        "./data/transcripts",
        "./data/processed_data",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.info("Creating .env file...")
        
        env_content = """# MyMind Therapeutic AI Configuration

# Database Configuration
DATABASE_URL=sqlite:///./data/mymind.db

# OpenAI Configuration (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Configuration (Required for speaker diarization)
HF_TOKEN=your_huggingface_token_here

# Audio Processing
AUDIO_UPLOAD_PATH=./data/raw_audio
TRANSCRIPT_PATH=./data/transcripts
PROCESSED_DATA_PATH=./data/processed_data

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Security
SECRET_KEY=your_secret_key_here_change_in_production
ALGORITHM=HS256

# Model Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1

# Processing Configuration
CHUNK_SIZE=3
BATCH_SIZE=4
MAX_SPEAKERS=5
"""
        
        with open(env_file, "w") as f:
            f.write(env_content)
        
        logger.info("Created .env file. Please update with your API keys!")
    else:
        logger.info(".env file already exists")

def main():
    """Main startup function"""
    
    logger.info("🧠 Starting MyMind Therapeutic AI System...")
    
    # Check Python version
    check_python_version()
    
    # Set up directories
    setup_directories()
    
    # Create environment file
    create_env_file()
    
    # Install core dependencies
    install_dependencies()
    
    logger.info("✅ Setup complete!")
    logger.info("")
    logger.info("🚀 To start the system:")
    logger.info("   python minimal_app.py")
    logger.info("")
    logger.info("🌐 Then visit: http://localhost:8000")
    logger.info("")
    logger.info("📖 For full functionality, install ML dependencies:")
    logger.info("   pip install -r requirements.txt")
    logger.info("")
    logger.info("🔑 Don't forget to add your API keys to .env file!")

if __name__ == "__main__":
    main()