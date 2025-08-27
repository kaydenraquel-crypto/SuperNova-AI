#!/bin/bash

# SuperNova AI - One-Click Setup Script for macOS/Linux
# Enterprise Financial Intelligence Platform

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP $1/8]${NC} $2"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2
}

# Function to get Node version
get_node_version() {
    node --version 2>/dev/null | cut -d'v' -f2 | cut -d'.' -f1
}

# Clear screen and show header
clear
echo -e "${WHITE}"
echo "========================================"
echo "  SuperNova AI - One-Click Setup"
echo "  Enterprise Financial Intelligence Platform"
echo "========================================"
echo -e "${NC}"

print_info "Starting SuperNova AI setup..."
echo

# Check if running with proper permissions
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. This is not recommended for development setup."
fi

# Step 1: Check Python installation
print_step 1 "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(get_python_version)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 11 ]]; then
        print_success "Found Python $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
        print_warning "Found Python $PYTHON_VERSION (recommended: 3.11+)"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        echo "Please install Python 3.11+ from https://python.org"
        exit 1
    fi
elif command_exists python; then
    print_warning "Using 'python' command (prefer python3)"
    PYTHON_CMD="python"
else
    print_error "Python not found!"
    echo "Please install Python 3.11+ from:"
    echo "  - macOS: brew install python@3.11 or download from python.org"
    echo "  - Linux: sudo apt install python3.11 python3.11-venv python3.11-pip"
    exit 1
fi

# Step 2: Check Node.js installation
print_step 2 "Checking Node.js installation..."
if command_exists node; then
    NODE_VERSION=$(get_node_version)
    if [[ $NODE_VERSION -ge 18 ]]; then
        print_success "Found Node.js v$NODE_VERSION"
    elif [[ $NODE_VERSION -ge 16 ]]; then
        print_warning "Found Node.js v$NODE_VERSION (recommended: v18+)"
    else
        print_error "Node.js 16+ required, found v$NODE_VERSION"
        echo "Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi
else
    print_error "Node.js not found!"
    echo "Please install Node.js 18+ from:"
    echo "  - macOS: brew install node or download from nodejs.org"
    echo "  - Linux: sudo apt install nodejs npm or use NodeSource repository"
    exit 1
fi

# Step 3: Check Git installation
print_step 3 "Checking Git installation..."
if command_exists git; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_success "Found Git $GIT_VERSION"
else
    print_warning "Git not found. Some features may not work properly."
fi

# Step 4: Create virtual environment
print_step 4 "Setting up Python virtual environment..."
if [[ -d ".venv" ]]; then
    print_info "Virtual environment already exists, activating..."
else
    print_info "Creating new virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [[ $? -ne 0 ]]; then
        print_error "Failed to create virtual environment"
        echo "Try: $PYTHON_CMD -m pip install --user virtualenv"
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate
print_success "Virtual environment activated"

# Step 5: Install Python dependencies
print_step 5 "Installing Python dependencies..."
print_info "This may take a few minutes..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
if [[ $? -ne 0 ]]; then
    print_error "Failed to install Python dependencies"
    echo "Try running: pip install --upgrade pip setuptools wheel"
    echo "Then run this script again"
    exit 1
fi
print_success "Python dependencies installed"

# Step 6: Install frontend dependencies
print_step 6 "Installing frontend dependencies..."
cd frontend

if [[ ! -d "node_modules" ]]; then
    print_info "Installing Node.js packages... (this may take a while)"
    npm install
    if [[ $? -ne 0 ]]; then
        print_error "Failed to install frontend dependencies"
        print_info "Trying with --legacy-peer-deps..."
        npm install --legacy-peer-deps
        if [[ $? -ne 0 ]]; then
            cd ..
            exit 1
        fi
    fi
else
    print_info "Node modules already exist, checking for updates..."
    npm update
fi

cd ..
print_success "Frontend dependencies installed"

# Step 7: Setup environment configuration
print_step 7 "Configuring environment..."
if [[ ! -f ".env" ]]; then
    print_info "Creating environment configuration..."
    cp .env.example .env
    echo
    echo -e "${YELLOW}[IMPORTANT]${NC} Please edit the .env file to configure:"
    echo "- API keys for LLM providers (OpenAI, Anthropic)"
    echo "- Database connections"
    echo "- Security settings"
    echo
    print_info "Default configuration will work for local development"
else
    print_info "Environment file already exists"
fi

# Step 8: Initialize database
print_step 8 "Initializing database..."
print_info "Setting up local SQLite database..."

python -c "
try:
    from supernova.db import init_db
    init_db()
    print('[OK] Database initialized successfully')
except Exception as e:
    print(f'[ERROR] Database initialization failed: {str(e)}')
    print('[INFO] You may need to configure database settings in .env')
" 2>/dev/null

echo
echo -e "${WHITE}"
echo "========================================"
echo "  SuperNova AI Setup Complete!"
echo "========================================"
echo -e "${NC}"

# Create start scripts
print_info "Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "Starting SuperNova AI Backend..."
source .venv/bin/activate
uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081
EOF

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "Starting SuperNova AI Frontend..."
cd frontend
npm start
EOF

# Combined startup script
cat > start_supernova.sh << 'EOF'
#!/bin/bash

echo
echo "======================================="
echo "  SuperNova AI - Starting All Services"
echo "======================================="
echo

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Check if ports are available
if check_port 8081; then
    echo "[WARN] Port 8081 is already in use"
fi

if check_port 3000; then
    echo "[WARN] Port 3000 is already in use"
fi

echo "[INFO] Starting backend server..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && ./start_backend.sh"'
else
    # Linux
    if command -v gnome-terminal >/dev/null; then
        gnome-terminal -- bash -c "cd $(pwd) && ./start_backend.sh; exec bash"
    elif command -v xterm >/dev/null; then
        xterm -e "cd $(pwd) && ./start_backend.sh; exec bash" &
    else
        echo "[INFO] Please run './start_backend.sh' in a separate terminal"
    fi
fi

sleep 5

echo "[INFO] Starting frontend server..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && ./start_frontend.sh"'
else
    # Linux
    if command -v gnome-terminal >/dev/null; then
        gnome-terminal -- bash -c "cd $(pwd) && ./start_frontend.sh; exec bash"
    elif command -v xterm >/dev/null; then
        xterm -e "cd $(pwd) && ./start_frontend.sh; exec bash" &
    else
        echo "[INFO] Please run './start_frontend.sh' in a separate terminal"
    fi
fi

echo
echo "[SUCCESS] SuperNova AI is starting!"
echo
echo "Backend:  http://localhost:8081"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8081/docs"
echo

# Wait a bit and try to open browser
sleep 3
if command -v open >/dev/null; then
    # macOS
    open http://localhost:3000
elif command -v xdg-open >/dev/null; then
    # Linux
    xdg-open http://localhost:3000
fi
EOF

# Make scripts executable
chmod +x start_backend.sh
chmod +x start_frontend.sh
chmod +x start_supernova.sh

print_success "Setup complete! Here's how to start SuperNova AI:"
echo
echo "  Option 1 - Start Everything:     ./start_supernova.sh"
echo "  Option 2 - Backend Only:         ./start_backend.sh"
echo "  Option 3 - Frontend Only:        ./start_frontend.sh"
echo
echo "  Access URLs:"
echo "  - Main Application: http://localhost:3000"
echo "  - API Documentation: http://localhost:8081/docs"
echo "  - Backend API: http://localhost:8081"
echo

print_info "First time? Edit .env file to add your API keys!"
echo

# Ask if user wants to start now
echo -n "Start SuperNova AI now? (y/n): "
read -r START_NOW

if [[ $START_NOW =~ ^[Yy]$ ]]; then
    print_info "Starting SuperNova AI..."
    ./start_supernova.sh
else
    print_info "You can start SuperNova AI later using ./start_supernova.sh"
fi

echo
echo "Thank you for using SuperNova AI! 🚀"