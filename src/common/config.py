from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # Database Configuration
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/mymind_db",
        env="DATABASE_URL"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Google Gemini Configuration
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # Hugging Face Configuration
    hf_token: str = Field(default="", env="HF_TOKEN")
    
    # Audio Processing Paths
    audio_upload_path: str = Field(
        default="./data/raw_audio",
        env="AUDIO_UPLOAD_PATH"
    )
    transcript_path: str = Field(
        default="./data/transcripts",
        env="TRANSCRIPT_PATH"
    )
    processed_data_path: str = Field(
        default="./data/processed_data",
        env="PROCESSED_DATA_PATH"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(
        default="your_secret_key_here",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # Model Configuration
    whisper_model: str = Field(default="large-v3", env="WHISPER_MODEL")
    whisper_device: str = Field(default="cuda", env="WHISPER_DEVICE")
    pyannote_model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        env="PYANNOTE_MODEL"
    )
    
    # Processing Configuration
    chunk_size: int = Field(default=3, env="CHUNK_SIZE")
    batch_size: int = Field(default=24, env="BATCH_SIZE")
    max_speakers: int = Field(default=10, env="MAX_SPEAKERS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields to prevent validation errors

# Global settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.audio_upload_path,
        settings.transcript_path,
        settings.processed_data_path,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories on import
ensure_directories()