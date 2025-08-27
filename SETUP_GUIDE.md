# SuperNova AI - Setup Guide

## 🚀 One-Click Installation Options

SuperNova AI provides multiple installation methods to suit different user preferences and technical expertise levels.

### Option 1: Windows Users (Recommended)
**Double-click to run:**
```
setup.bat
```

**What it does:**
- ✅ Automatically checks Python 3.11+ and Node.js 18+
- ✅ Creates virtual environment and installs dependencies
- ✅ Sets up frontend React application
- ✅ Initializes SQLite database
- ✅ Creates convenient startup scripts
- ✅ Provides colored output and progress indicators

**After setup, start SuperNova AI:**
```
start_supernova.bat    (Starts both backend and frontend)
start_backend.bat      (Backend API only)
start_frontend.bat     (Frontend UI only)
```

### Option 2: macOS/Linux Users (Recommended)
**Run in terminal:**
```bash
chmod +x setup.sh
./setup.sh
```

**What it does:**
- ✅ Cross-platform compatibility (macOS, Ubuntu, CentOS, etc.)
- ✅ Intelligent dependency detection and version checking
- ✅ Automatic virtual environment setup
- ✅ Colored terminal output with progress indicators
- ✅ Creates executable startup scripts
- ✅ Opens browser automatically when ready

**After setup, start SuperNova AI:**
```bash
./start_supernova.sh    # Starts both backend and frontend
./start_backend.sh      # Backend API only
./start_frontend.sh     # Frontend UI only
```

### Option 3: GUI Setup Wizard (All Platforms)
**For users who prefer graphical interface:**
```bash
python setup_wizard.py
```

**Features:**
- 🖥️ Cross-platform GUI with step-by-step wizard
- ⚙️ Visual prerequisite checking
- 📝 Interactive configuration editor
- 📊 Real-time installation progress
- 🔧 Environment variable management
- 🌐 Built-in browser integration

**Requirements:** Python with tkinter (usually included)

### Option 4: Docker (Advanced Users)
**For containerized deployment:**
```bash
# Development
docker-compose -f deploy/docker/docker-compose.development.yml up -d

# Production
docker-compose -f deploy/docker/docker-compose.production.yml up -d
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+ (3.11+ recommended)
- **Node.js**: 16+ (18+ recommended)
- **RAM**: 4GB
- **Disk**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.11+
- **Node.js**: 18+
- **RAM**: 8GB
- **Disk**: 4GB free space
- **Git**: Latest version (for development)

## 🔧 Manual Installation (Advanced)

If you prefer manual setup or encounter issues:

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/supernova-ai.git
cd supernova-ai
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 4. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Database Initialization
```bash
python -c "from supernova.db import init_db; init_db()"
```

### 6. Start Services
```bash
# Terminal 1 - Backend
uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081

# Terminal 2 - Frontend
cd frontend && npm start
```

## 🌐 Access URLs

After successful setup:

- **Main Application**: http://localhost:3000
- **API Documentation**: http://localhost:8081/docs
- **Backend API**: http://localhost:8081
- **WebSocket**: ws://localhost:8081/ws

## ⚙️ Configuration

### Environment Variables (.env file)

**Essential Configuration:**
```bash
# LLM Provider API Keys (Optional for testing)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Database (SQLite default, PostgreSQL for production)
DATABASE_URL=sqlite:///supernova.db

# Security (Auto-generated if empty)
JWT_SECRET_KEY=your-secret-key-here

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

**Advanced Configuration:**
```bash
# Redis (Optional, for caching)
REDIS_URL=redis://localhost:6379

# TimescaleDB (Optional, for time-series data)
TIMESCALE_URL=postgresql://user:pass@localhost/timescale

# Email (Optional, for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# External APIs (Optional)
ALPHA_VANTAGE_API_KEY=your-av-key
POLYGON_API_KEY=your-polygon-key
```

## 🔍 Troubleshooting

### Common Issues

**Python not found:**
- Windows: Install from https://python.org, check "Add to PATH"
- macOS: `brew install python@3.11`
- Linux: `sudo apt install python3.11 python3.11-venv`

**Node.js not found:**
- Windows: Install from https://nodejs.org
- macOS: `brew install node`
- Linux: `sudo apt install nodejs npm`

**Permission denied (macOS/Linux):**
```bash
chmod +x setup.sh
chmod +x start_supernova.sh
```

**Port already in use:**
```bash
# Kill processes on ports 3000 and 8081
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:3000 | xargs kill -9
lsof -ti:8081 | xargs kill -9
```

**Database initialization fails:**
- Check .env DATABASE_URL setting
- Ensure write permissions in project directory
- For PostgreSQL, verify database exists and credentials

**Frontend build fails:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

**Import errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version`

### Getting Help

1. **Check logs**: Look for error messages in terminal output
2. **Review documentation**: Check docs/ folder for detailed guides
3. **Environment file**: Verify .env configuration
4. **Dependencies**: Ensure all requirements are installed
5. **Permissions**: Check file and directory permissions

## 📊 Verification

After setup, verify installation:

### Health Check
```bash
curl http://localhost:8081/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "2.0.0",
  "components": {
    "database": "ok",
    "cache": "ok",
    "api": "ok"
  }
}
```

### Frontend Check
Open http://localhost:3000 and verify:
- ✅ SuperNova AI dashboard loads
- ✅ Material-UI theme applied
- ✅ Navigation menu responsive
- ✅ Chat interface accessible

### API Documentation
Open http://localhost:8081/docs and verify:
- ✅ Interactive API documentation loads
- ✅ All endpoints listed
- ✅ Authentication endpoints available
- ✅ WebSocket documentation present

## 🎉 Success!

If all checks pass, SuperNova AI is ready! 

**Next steps:**
1. Add your API keys to .env file
2. Explore the dashboard at http://localhost:3000
3. Try the AI chat interface
4. Review documentation in docs/ folder
5. Check out example portfolios and strategies

**Welcome to SuperNova AI - Enterprise Financial Intelligence Platform!** 🚀