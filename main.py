#!/usr/bin/env python3
"""
SuperNova Financial Advisor API Server
Main application entry point
"""

import uvicorn
from supernova.api import app

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8081, 
        reload=True,
        log_level="info"
    )