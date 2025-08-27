# Developer Setup Guide

This guide will help developers set up a local development environment for SuperNova AI. Follow these steps to get the platform running on your local machine for development and testing.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Database Configuration](#database-configuration)
6. [External Services](#external-services)
7. [Development Workflow](#development-workflow)
8. [Testing Setup](#testing-setup)
9. [Debugging and Tools](#debugging-and-tools)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Development Requirements:**
- **OS**: Windows 10+, macOS 10.15+, or Linux Ubuntu 18.04+
- **Memory**: 16GB RAM (32GB recommended for ML workloads)
- **Storage**: 50GB available disk space
- **CPU**: 4+ cores recommended
- **Network**: Stable broadband connection

### Required Software

#### Core Development Tools
- **Python 3.9+**: Primary backend language
- **Node.js 16.0+**: Frontend development and build tools
- **Git**: Version control system
- **Docker**: Containerization (recommended)
- **PostgreSQL 13+**: Production database (optional for development)

#### Development Environment
- **IDE/Editor**: VS Code, PyCharm, or similar
- **Terminal**: Command line access
- **Web Browser**: Chrome, Firefox, or Safari (latest version)

### Package Managers
- **pip**: Python package management
- **npm/yarn**: Node.js package management
- **pipenv** or **poetry**: Python virtual environment (optional)

## Environment Setup

### 1. Clone Repository

```bash
# Clone the SuperNova AI repository
git clone https://github.com/supernova-ai/supernova.git
cd supernova

# Create development branch
git checkout -b feature/development-setup
```

### 2. Environment Variables

Create environment configuration files:

```bash
# Copy environment template
cp .env.example .env

# Copy frontend environment template
cp frontend/.env.example frontend/.env
```

Edit `.env` with your development settings:

```bash
# Development Configuration
NODE_ENV=development
DEBUG=true
SUPERNOVA_HOST=localhost
SUPERNOVA_PORT=8081
FRONTEND_PORT=3000

# Database Configuration
DATABASE_URL=sqlite:///./supernova.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost:5432/supernova

# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your-secret-key-for-development-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-for-development
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# LLM API Keys (Optional - for AI features)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# External APIs
NOVASIGNAL_API_KEY=your_novasignal_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# TimescaleDB Configuration (Optional)
TIMESCALE_ENABLED=false
TIMESCALE_URL=postgresql://user:password@localhost:5432/timescaledb

# Development Features
ENABLE_DEBUG_TOOLBAR=true
ENABLE_WEBSOCKETS=true
WEBSOCKET_PORT=8082
CORS_ORIGINS=["http://localhost:3000"]

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
```

### 3. Development Tools Installation

#### Install Development Dependencies

**macOS (using Homebrew):**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install core tools
brew install python@3.9 node@16 postgresql@13 redis git

# Install optional tools
brew install --cask docker
brew install --cask visual-studio-code
```

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update

# Install Python and Node.js
sudo apt install python3.9 python3.9-venv python3-pip
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install nodejs

# Install PostgreSQL and Redis
sudo apt install postgresql-13 postgresql-contrib redis-server

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Git
sudo apt install git
```

**Windows:**
```bash
# Using Chocolatey package manager
# Install Chocolatey first: https://chocolatey.org/install

# Install core tools
choco install python3 nodejs postgresql redis git docker-desktop vscode

# Alternatively, download and install manually:
# - Python: https://python.org/downloads
# - Node.js: https://nodejs.org/download
# - PostgreSQL: https://postgresql.org/download
# - Git: https://git-scm.com/download
```

## Backend Setup

### 1. Python Environment Setup

#### Create Virtual Environment

```bash
# Navigate to project root
cd supernova

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

**Development Dependencies (`requirements-dev.txt`):**
```text
# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Code Quality
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
isort>=5.12.0
pre-commit>=3.3.0

# Development Tools
ipython>=8.14.0
jupyter>=1.0.0
httpx>=0.24.0  # For API testing

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.1.0

# Performance
line-profiler>=4.0.0
memory-profiler>=0.61.0
```

### 2. Backend Configuration

#### Initialize Database

```bash
# Initialize SQLite database (development)
python -c "from supernova.db import init_db; init_db()"

# Create sample data (optional)
python -c "from supernova.db import seed_database; seed_database()"

# For PostgreSQL setup:
# createdb supernova
# python scripts/migrate_database.py
```

#### Verify Backend Installation

```bash
# Run basic health check
python -c "import supernova; print('Backend setup successful')"

# Test database connection
python -c "from supernova.db import SessionLocal; db = SessionLocal(); print('Database connection successful'); db.close()"

# Test API startup (without starting server)
python -c "from supernova.api import app; print('API initialization successful')"
```

### 3. Start Backend Development Server

```bash
# Start with hot reloading
uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081

# Or using Python directly
python -m uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081

# For production-like testing
gunicorn supernova.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8081
```

**Verify Backend is Running:**
- API Documentation: http://localhost:8081/docs
- Health Check: http://localhost:8081/health
- Alternative Docs: http://localhost:8081/redoc

## Frontend Setup

### 1. Node.js Environment Setup

```bash
# Navigate to frontend directory
cd frontend

# Verify Node.js version
node --version  # Should be 16.0.0 or higher
npm --version   # Should be 8.0.0 or higher

# Install dependencies
npm install

# Or using Yarn (if preferred)
# yarn install
```

### 2. Frontend Configuration

Edit `frontend/.env`:

```bash
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8081
REACT_APP_WS_BASE_URL=ws://localhost:8082

# Development Configuration
GENERATE_SOURCEMAP=true
REACT_APP_ENVIRONMENT=development

# Feature Flags
REACT_APP_ENABLE_DEBUG=true
REACT_APP_ENABLE_MOCK_DATA=false

# External Services
REACT_APP_SENTRY_DSN=your_sentry_dsn_here
REACT_APP_GOOGLE_ANALYTICS_ID=your_ga_id_here
```

### 3. Development Dependencies

The frontend uses these key development tools:

**Build Tools:**
- **Webpack**: Module bundler
- **Babel**: JavaScript compiler  
- **TypeScript**: Type checking
- **ESLint**: Code linting
- **Prettier**: Code formatting

**Testing Tools:**
- **Jest**: Testing framework
- **React Testing Library**: Component testing
- **MSW**: API mocking for tests

**Development Server:**
```bash
# Start development server with hot reloading
npm start

# Or specify port
PORT=3001 npm start

# Build for production testing
npm run build

# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

### 4. Verify Frontend Setup

**Check Development Server:**
- Frontend URL: http://localhost:3000
- Hot reloading should work when you edit files
- Browser console should show no errors

**Test Build Process:**
```bash
# Test production build
npm run build

# Serve production build locally
npx serve -s build -l 3001
```

## Database Configuration

### SQLite (Development Default)

SQLite requires no additional setup and is used by default for development:

```python
# Verify SQLite setup
python -c "
from supernova.db import SessionLocal, init_db
init_db()
db = SessionLocal()
print(f'Database tables created: {len(db.get_bind().table_names())}')
db.close()
"
```

### PostgreSQL (Production-like Development)

#### 1. Install and Configure PostgreSQL

```bash
# Start PostgreSQL service
# macOS (Homebrew):
brew services start postgresql

# Ubuntu/Debian:
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Windows:
# Start PostgreSQL service from Services panel
```

#### 2. Create Development Database

```bash
# Connect as postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE supernova_dev;
CREATE USER supernova_dev WITH PASSWORD 'dev_password';
GRANT ALL PRIVILEGES ON DATABASE supernova_dev TO supernova_dev;

# Exit PostgreSQL
\q
```

#### 3. Update Environment Configuration

```bash
# Update .env file
DATABASE_URL=postgresql://supernova_dev:dev_password@localhost:5432/supernova_dev
```

#### 4. Initialize PostgreSQL Database

```bash
# Run database migrations
python scripts/init_postgres.py

# Verify connection
python -c "
from supernova.db import SessionLocal
db = SessionLocal()
result = db.execute('SELECT version()')
print(f'PostgreSQL version: {result.fetchone()[0]}')
db.close()
"
```

### TimescaleDB (Optional - for Time Series Data)

#### 1. Install TimescaleDB

```bash
# macOS (Homebrew):
brew tap timescale/tap
brew install timescaledb
timescaledb-tune

# Ubuntu/Debian:
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update
sudo apt install timescaledb-2-postgresql-13
```

#### 2. Configure TimescaleDB

```bash
# Add to postgresql.conf
sudo echo "shared_preload_libraries = 'timescaledb'" >> /etc/postgresql/13/main/postgresql.conf

# Restart PostgreSQL
sudo systemctl restart postgresql

# Create TimescaleDB extension
psql -U supernova_dev -d supernova_dev -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

#### 3. Update Environment for TimescaleDB

```bash
# Update .env file
TIMESCALE_ENABLED=true
TIMESCALE_URL=postgresql://supernova_dev:dev_password@localhost:5432/supernova_dev
```

## External Services

### LLM API Setup

#### OpenAI Setup

1. Visit [platform.openai.com](https://platform.openai.com)
2. Create account and generate API key
3. Add to `.env`:
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

#### Anthropic Claude Setup

1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Create account and generate API key
3. Add to `.env`:
   ```bash
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ```

#### Test LLM Integration

```bash
# Test OpenAI integration
python -c "
from supernova.advisor import test_llm_integration
result = test_llm_integration('openai')
print(f'OpenAI test: {result}')
"

# Test Anthropic integration
python -c "
from supernova.advisor import test_llm_integration
result = test_llm_integration('anthropic')
print(f'Anthropic test: {result}')
"
```

### Market Data APIs

#### Alpha Vantage Setup

1. Visit [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Get free API key
3. Add to `.env`:
   ```bash
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
   ```

#### NovaSignal Integration

1. Contact NovaSignal for API credentials
2. Add to `.env`:
   ```bash
   NOVASIGNAL_API_KEY=your-novasignal-key
   NOVASIGNAL_BASE_URL=https://api.novasignal.com
   ```

## Development Workflow

### 1. Daily Development Setup

```bash
# Start development session
cd supernova

# Activate Python environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Start backend server (Terminal 1)
uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081

# Start frontend server (Terminal 2)
cd frontend
npm start

# Optional: Start additional services (Terminal 3)
redis-server  # If using Redis
```

### 2. Development Commands

#### Backend Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=supernova --cov-report=html

# Run specific test file
pytest tests/test_advisor.py

# Run linting
flake8 supernova/
black supernova/
mypy supernova/

# Database operations
python -m alembic revision --autogenerate -m "Description"
python -m alembic upgrade head

# Interactive Python shell with app context
python -i -c "from supernova import *"
```

#### Frontend Development

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Linting and formatting
npm run lint
npm run lint:fix
npm run format

# Type checking
npm run type-check

# Bundle analysis
npm run analyze
```

### 3. Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature description"

# Push and create pull request
git push origin feature/your-feature-name

# Keep branch updated
git checkout main
git pull origin main
git checkout feature/your-feature-name
git merge main
```

#### Commit Message Format

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tool changes

## Testing Setup

### Backend Testing

#### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_advisor.py
│   ├── test_backtester.py
│   └── test_strategy_engine.py
├── integration/          # Integration tests
│   ├── test_api_endpoints.py
│   └── test_database.py
├── e2e/                 # End-to-end tests
│   └── test_full_workflow.py
└── fixtures/            # Test data and fixtures
    ├── sample_data.py
    └── mock_responses.json
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=supernova --cov-report=html
open htmlcov/index.html  # View coverage report

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_advisor.py::test_risk_scoring
```

#### Test Configuration

**pytest.ini:**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=supernova
    --cov-branch
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    external: Tests requiring external services
```

### Frontend Testing

#### Test Structure

```
src/
├── components/
│   └── __tests__/       # Component tests
├── hooks/
│   └── __tests__/       # Hook tests
├── pages/
│   └── __tests__/       # Page tests
├── services/
│   └── __tests__/       # Service tests
└── __tests__/          # General tests
```

#### Running Frontend Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run tests in CI mode
CI=true npm test

# Update test snapshots
npm test -- --updateSnapshot
```

#### Jest Configuration

**jest.config.js:**
```javascript
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/reportWebVitals.ts'
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  }
};
```

## Debugging and Tools

### Backend Debugging

#### VS Code Configuration

**.vscode/launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/.venv/bin/uvicorn",
      "args": [
        "supernova.api:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8081"
      ],
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    }
  ]
}
```

#### Debugging Tools

```bash
# Interactive debugging with IPython
pip install ipython
ipython

# Debug with pdb
python -m pdb script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Line profiling
pip install line-profiler
kernprof -l -v script.py
```

### Frontend Debugging

#### Browser DevTools Setup

1. **Chrome DevTools:**
   - Install React Developer Tools extension
   - Install Redux DevTools extension (if using Redux)

2. **VS Code Extensions:**
   - ES7+ React/Redux/React-Native snippets
   - TypeScript Importer
   - Prettier - Code formatter
   - ESLint

#### Debugging Configuration

**VS Code Browser Debugging:**
```json
{
  "name": "Launch Chrome",
  "type": "pwa-chrome",
  "request": "launch",
  "url": "http://localhost:3000",
  "webRoot": "${workspaceFolder}/frontend/src",
  "sourceMapPathOverrides": {
    "webpack:///src/*": "${webRoot}/*"
  }
}
```

### Database Debugging

#### SQL Query Logging

```python
# Enable SQL logging in development
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

#### Database Tools

- **pgAdmin**: PostgreSQL administration (GUI)
- **DBeaver**: Universal database tool
- **SQLite Browser**: SQLite database browser

### API Testing Tools

#### HTTPie Examples

```bash
# Install HTTPie
pip install httpie

# Test API endpoints
http GET localhost:8081/docs
http POST localhost:8081/intake name="Test User" email="test@example.com"
http POST localhost:8081/advice profile_id:=1 symbol="AAPL" bars:='[{"timestamp":"2025-08-26T09:30:00Z","open":150.50,"high":152.75,"low":149.25,"close":151.80,"volume":1250000}]'
```

#### Postman Collection

Import the SuperNova API Postman collection for comprehensive API testing:

1. Download collection from `/docs/api/supernova-postman-collection.json`
2. Import into Postman
3. Set environment variables for API base URL and tokens

## Troubleshooting

### Common Issues

#### Backend Issues

**Issue: Import Errors**
```bash
# Solution: Check Python path and virtual environment
echo $PYTHONPATH
which python
pip list
```

**Issue: Database Connection Errors**
```bash
# SQLite: Check file permissions
ls -la supernova.db

# PostgreSQL: Check service status
sudo systemctl status postgresql
```

**Issue: API Server Won't Start**
```bash
# Check port availability
lsof -i :8081
netstat -an | grep 8081

# Check logs for errors
tail -f logs/supernova.log
```

#### Frontend Issues

**Issue: Node Modules Issues**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Issue: Build Failures**
```bash
# Clear build cache
npm run clean
rm -rf build/

# Check TypeScript errors
npm run type-check
```

**Issue: Hot Reloading Not Working**
```bash
# Check file system watch limits (Linux)
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### Environment Issues

**Issue: Port Conflicts**
```bash
# Find process using port
lsof -i :8081
kill -9 <PID>

# Use alternative ports
PORT=3001 npm start
uvicorn supernova.api:app --port 8082
```

**Issue: Permission Errors**
```bash
# Fix file permissions
chmod +x scripts/*.py
chown -R $USER:$USER .

# Fix Python virtual environment permissions
chmod -R 755 .venv/
```

### Getting Help

#### Development Resources

1. **Documentation**: Check `/docs` directory for detailed guides
2. **Code Examples**: Review `/examples` directory for usage patterns
3. **Test Cases**: Study test files for expected behavior
4. **API Documentation**: Use `/docs` endpoint for interactive API testing

#### Community Support

1. **GitHub Issues**: Report bugs and request features
2. **Discussion Forums**: Ask questions and share solutions  
3. **Slack Channel**: Real-time developer chat
4. **Stack Overflow**: Tag questions with `supernova-ai`

#### Professional Support

- **Email**: developer-support@supernova-ai.com
- **Documentation**: https://docs.supernova-ai.com
- **Status Page**: https://status.supernova-ai.com

---

**Next Steps:**
- [Architecture Overview](architecture.md)
- [API Reference](../api-reference/README.md)
- [Contributing Guide](contributing.md)
- [Testing Guide](testing.md)

Last Updated: August 26, 2025  
Version: 1.0.0