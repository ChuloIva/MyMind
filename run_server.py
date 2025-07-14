#!/usr/bin/env python3
"""
MyMind Server Launcher
Starts the FastAPI server for the MyMind Therapeutic AI Platform
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the MyMind server"""
    print("🧠 Starting MyMind Therapeutic AI Platform...")
    print("=" * 50)
    
    try:
        # Test imports
        print("Testing imports...")
        from src.database.database import engine
        from src.database.models import NeedCategory, LifeSegment
        print("✅ Database imports successful")
        
        from src.preprocessing.llm_processing.keyword_extraction import KeywordExtractor
        print("✅ Keyword extraction available")
        
        from src.analysis.therapeutic_methods.distortions import CognitiveDistortionAnalyzer
        print("✅ CBT analysis available")
        
        try:
            from src.preprocessing.llm_processing.needs_extraction import NeedsExtractor
            print("✅ Needs extraction available")
        except Exception as e:
            print(f"⚠️  Needs extraction may have issues: {e}")
        
        # Try to import the main app
        from src.api.main import app
        print("✅ FastAPI app imported successfully")
        
        print("\n🚀 Starting server on http://localhost:8000")
        print("📱 Open your browser and navigate to http://localhost:8000")
        print("📖 API docs available at http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that all required dependencies are installed")
        print("3. Verify database is initialized: python init_db.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()