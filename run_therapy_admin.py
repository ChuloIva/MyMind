#!/usr/bin/env python3
"""
Standalone server for MyMind Therapy Admin
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🚀 Starting MyMind Therapy Admin Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 Admin UI: http://localhost:8000")
    print("📊 Analysis UI: http://localhost:8000/analysis")
    print("🔧 API Health: http://localhost:8000/health")
    print("📚 API Docs: http://localhost:8000/docs")
    print()
    
    try:
        uvicorn.run(
            "src.api.main_simple:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)