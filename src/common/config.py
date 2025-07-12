from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql://postgres:password@localhost:5432/mymind_db"
    
    # OpenAI Configuration
    openai_api_key: str = ""
    
    # Hugging Face Configuration
    hf_token: str = ""
    
    # Audio Processing Paths
    audio_upload_path: str = "./data/raw_audio"
    transcript_path: str = "./data/transcripts"
    processed_data_path: str = "./data/processed_data"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Security
    secret_key: str = "your_secret_key_here"
    algorithm: str = "HS256"
    
    # Model Configuration
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    
    # Processing Configuration
    chunk_size: int = 3
    batch_size: int = 24
    max_speakers: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.audio_upload_path,
        settings.transcript_path,
        settings.processed_data_path,
        "./data",
        "./data/raw_audio",
        "./data/transcripts", 
        "./data/processed_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories on import
ensure_directories()