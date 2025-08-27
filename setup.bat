@echo off
echo.
echo ========================================
echo  SuperNova AI - One-Click Setup
echo  Enterprise Financial Intelligence Platform
echo ========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with administrator privileges...
) else (
    echo [WARN] Not running as administrator. Some features may require elevation.
)

:: Set colors for better output
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo [INFO] Starting SuperNova AI setup...
echo.

:: Check Python installation
echo [STEP 1/8] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.11+ from https://python.org
    echo [INFO] Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

:: Check Node.js installation
echo [STEP 2/8] Checking Node.js installation...
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Node.js not found! Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

for /f %%i in ('node --version') do set NODE_VERSION=%%i
echo [OK] Found Node.js %NODE_VERSION%

:: Check if Git is available
echo [STEP 3/8] Checking Git installation...
git --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] Git not found. Some features may not work properly.
) else (
    for /f "tokens=3" %%i in ('git --version') do set GIT_VERSION=%%i
    echo [OK] Found Git %GIT_VERSION%
)

:: Create virtual environment
echo [STEP 4/8] Setting up Python virtual environment...
if exist .venv (
    echo [INFO] Virtual environment already exists, activating...
) else (
    echo [INFO] Creating new virtual environment...
    python -m venv .venv
    if %errorLevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: Install Python dependencies
echo [STEP 5/8] Installing Python dependencies...
echo [INFO] This may take a few minutes...
pip install --upgrade pip
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies
    echo [INFO] Try running: pip install --upgrade pip
    echo [INFO] Then run this script again
    pause
    exit /b 1
)
echo [OK] Python dependencies installed

:: Install frontend dependencies
echo [STEP 6/8] Installing frontend dependencies...
cd frontend
if not exist node_modules (
    echo [INFO] Installing Node.js packages... (this may take a while)
    npm install
    if %errorLevel% neq 0 (
        echo [ERROR] Failed to install frontend dependencies
        cd ..
        pause
        exit /b 1
    )
) else (
    echo [INFO] Node modules already exist, checking for updates...
    npm update
)
cd ..
echo [OK] Frontend dependencies installed

:: Setup environment configuration
echo [STEP 7/8] Configuring environment...
if not exist .env (
    echo [INFO] Creating environment configuration...
    copy .env.example .env
    echo.
    echo [IMPORTANT] Please edit the .env file to configure:
    echo - API keys for LLM providers (OpenAI, Anthropic)
    echo - Database connections
    echo - Security settings
    echo.
    echo [INFO] Default configuration will work for local development
) else (
    echo [INFO] Environment file already exists
)

:: Initialize database
echo [STEP 8/8] Initializing database...
echo [INFO] Setting up local SQLite database...
python -c "
try:
    from supernova.db import init_db
    init_db()
    print('[OK] Database initialized successfully')
except Exception as e:
    print(f'[ERROR] Database initialization failed: {str(e)}')
    print('[INFO] You may need to configure database settings in .env')
"

echo.
echo ========================================
echo  SuperNova AI Setup Complete!
echo ========================================
echo.

:: Create start scripts
echo [INFO] Creating startup scripts...

echo @echo off > start_backend.bat
echo echo Starting SuperNova AI Backend... >> start_backend.bat
echo call .venv\Scripts\activate.bat >> start_backend.bat
echo uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081 >> start_backend.bat

echo @echo off > start_frontend.bat
echo echo Starting SuperNova AI Frontend... >> start_frontend.bat
echo cd frontend >> start_frontend.bat
echo npm start >> start_frontend.bat

echo @echo off > start_supernova.bat
echo echo. >> start_supernova.bat
echo echo ======================================= >> start_supernova.bat
echo echo  SuperNova AI - Starting All Services >> start_supernova.bat
echo echo ======================================= >> start_supernova.bat
echo echo. >> start_supernova.bat
echo echo [INFO] Starting backend server... >> start_supernova.bat
echo start "SuperNova Backend" cmd /k start_backend.bat >> start_supernova.bat
echo timeout /t 5 /nobreak >> start_supernova.bat
echo echo [INFO] Starting frontend server... >> start_supernova.bat
echo start "SuperNova Frontend" cmd /k start_frontend.bat >> start_supernova.bat
echo echo. >> start_supernova.bat
echo echo [SUCCESS] SuperNova AI is starting! >> start_supernova.bat
echo echo. >> start_supernova.bat
echo echo Backend:  http://localhost:8081 >> start_supernova.bat
echo echo Frontend: http://localhost:3000 >> start_supernova.bat
echo echo API Docs: http://localhost:8081/docs >> start_supernova.bat
echo echo. >> start_supernova.bat
echo pause >> start_supernova.bat

echo [SUCCESS] Setup complete! Here's how to start SuperNova AI:
echo.
echo   Option 1 - Start Everything:     start_supernova.bat
echo   Option 2 - Backend Only:         start_backend.bat  
echo   Option 3 - Frontend Only:        start_frontend.bat
echo.
echo   Access URLs:
echo   - Main Application: http://localhost:3000
echo   - API Documentation: http://localhost:8081/docs
echo   - Backend API: http://localhost:8081
echo.
echo [INFO] First time? Edit .env file to add your API keys!
echo.

:: Ask if user wants to start now
set /p START_NOW="Start SuperNova AI now? (y/n): "
if /i "%START_NOW%"=="y" (
    echo [INFO] Starting SuperNova AI...
    start start_supernova.bat
) else (
    echo [INFO] You can start SuperNova AI later using start_supernova.bat
)

echo.
echo Thank you for using SuperNova AI! 🚀
pause