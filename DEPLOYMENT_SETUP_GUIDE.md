# SuperNova Conversational Agent - Production Deployment Guide

**Version**: 3.0 Extended Framework  
**Target Environment**: Production  
**Deployment Type**: Full Stack Conversational AI System  

---

## Pre-Deployment Checklist

### 🔴 **Critical Prerequisites**
- [ ] Fix import dependencies in core modules
- [ ] Implement user authentication system
- [ ] Configure LLM provider (OpenAI/Anthropic/etc.)
- [ ] Set up production database
- [ ] Configure environment variables

### 🟡 **High Priority**
- [ ] Implement API rate limiting
- [ ] Set up SSL/TLS certificates
- [ ] Configure monitoring and logging
- [ ] Set up backup and recovery systems
- [ ] Security hardening implementation

---

## Environment Setup

### 1. **System Requirements**
```bash
# Minimum Requirements
- OS: Linux (Ubuntu 20.04+) / Windows Server / macOS
- Python: 3.11+
- RAM: 4GB minimum, 8GB recommended
- Storage: 10GB minimum, 50GB recommended
- Network: Stable internet connection for LLM API calls

# Recommended Production Environment
- OS: Ubuntu 22.04 LTS
- Python: 3.11 or 3.12
- RAM: 16GB
- Storage: 100GB SSD
- CPU: 4+ cores
```

### 2. **Python Environment Setup**
```bash
# Create virtual environment
python -m venv supernova_env
source supernova_env/bin/activate  # Linux/macOS
# or
supernova_env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn uvicorn[standard] redis psycopg2-binary
```

### 3. **Database Setup**

#### **PostgreSQL (Recommended for Production)**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createdb supernova_db
sudo -u postgres createuser supernova_user
sudo -u postgres psql -c "ALTER USER supernova_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE supernova_db TO supernova_user;"
```

#### **SQLite (Development/Small Scale)**
```bash
# SQLite is included by default
# Database will be created automatically in supernova.db
```

### 4. **Redis Setup (Optional - for Caching)**
```bash
# Install Redis
sudo apt install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf
# Set: maxmemory 1gb
# Set: maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

---

## Environment Variables Configuration

### 1. **Create Production Environment File**
```bash
# Create .env file
cp .env.example .env
nano .env
```

### 2. **Critical Environment Variables**
```bash
# === DATABASE CONFIGURATION ===
DATABASE_URL=postgresql://supernova_user:secure_password@localhost/supernova_db
# For SQLite: DATABASE_URL=sqlite:///./supernova.db

# === LLM PROVIDER CONFIGURATION ===
# Choose ONE primary provider
LLM_PROVIDER=openai  # or anthropic, ollama, huggingface

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo

# Anthropic Configuration (alternative)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# LLM_MODEL=claude-3-sonnet

# === API CONFIGURATION ===
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false
LOG_LEVEL=INFO

# === SECURITY CONFIGURATION ===
SECRET_KEY=your_very_secure_secret_key_here_64_characters_minimum
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
ALLOWED_HOSTS=["your-domain.com", "api.your-domain.com"]

# === CORS CONFIGURATION ===
CORS_ORIGINS=["https://your-frontend-domain.com", "https://www.your-domain.com"]
CORS_CREDENTIALS=true

# === REDIS CONFIGURATION (Optional) ===
REDIS_URL=redis://localhost:6379/0

# === RATE LIMITING ===
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# === LOGGING ===
LOG_FILE=/var/log/supernova/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# === WEBSOCKET CONFIGURATION ===
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_MAX_CONNECTIONS=1000
```

### 3. **Generate Secure Keys**
```python
# Generate secure secret keys
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64))"
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))"
```

---

## Database Migration

### 1. **Initialize Database Schema**
```python
# Create database initialization script
# create_database.py

from supernova.db import Base, engine, SessionLocal
from supernova.config import settings

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

def create_default_data():
    """Create default data for the application"""
    db = SessionLocal()
    try:
        # Add any default data here
        # Example: default user profiles, system settings, etc.
        db.commit()
        print("✅ Default data created successfully")
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating default data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
    create_default_data()
```

```bash
# Run database initialization
python create_database.py
```

### 2. **Verify Database Connection**
```python
# test_database.py
from supernova.db import SessionLocal, User

def test_database():
    """Test database connection"""
    try:
        db = SessionLocal()
        # Simple query test
        user_count = db.query(User).count()
        print(f"✅ Database connection successful. Users: {user_count}")
        db.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    test_database()
```

---

## Authentication Implementation

### 1. **Create Authentication Module**
```python
# supernova/auth.py
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer
from .config import settings
from .db import SessionLocal, User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(security)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token.credentials, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    
    if user is None:
        raise credentials_exception
    return user
```

### 2. **Add Authentication Endpoints**
```python
# Add to supernova/api.py
from .auth import verify_password, get_password_hash, create_access_token, get_current_user

@app.post("/auth/register")
async def register(username: str, email: str, password: str):
    """Register new user"""
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username or email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"message": "User created successfully", "user_id": new_user.id}
    finally:
        db.close()

@app.post("/auth/login")
async def login(username: str, password: str):
    """Login user"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token = create_access_token(data={"sub": str(user.id)})
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()

@app.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }
```

---

## Rate Limiting Implementation

### 1. **Install Rate Limiting Library**
```bash
pip install slowapi
```

### 2. **Add Rate Limiting to API**
```python
# Add to supernova/api.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Create limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to chat endpoints
@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, ...):
    # Existing chat endpoint code
    pass

@app.websocket("/ws/chat")
@limiter.limit("1/second")
async def websocket_endpoint(websocket: WebSocket, ...):
    # Existing websocket code
    pass
```

---

## SSL/TLS Configuration

### 1. **Generate SSL Certificates**

#### **Using Let's Encrypt (Recommended)**
```bash
# Install Certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com -d api.your-domain.com

# Certificates will be in /etc/letsencrypt/live/your-domain.com/
```

#### **Self-Signed Certificates (Development)**
```bash
# Generate private key
openssl genrsa -out private_key.pem 2048

# Generate certificate signing request
openssl req -new -key private_key.pem -out csr.pem

# Generate self-signed certificate
openssl req -x509 -days 365 -key private_key.pem -in csr.pem -out certificate.pem
```

### 2. **Configure SSL in Production**
```python
# production_server.py
import uvicorn
from supernova.api import app

if __name__ == "__main__":
    uvicorn.run(
        "supernova.api:app",
        host="0.0.0.0",
        port=443,
        ssl_keyfile="/etc/letsencrypt/live/your-domain.com/privkey.pem",
        ssl_certfile="/etc/letsencrypt/live/your-domain.com/fullchain.pem",
        workers=4
    )
```

---

## Production Server Setup

### 1. **Systemd Service Configuration**
```bash
# Create service file
sudo nano /etc/systemd/system/supernova.service
```

```ini
[Unit]
Description=SuperNova Conversational Agent API
After=network.target postgresql.service

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/supernova
Environment="PATH=/opt/supernova/venv/bin"
ExecStart=/opt/supernova/venv/bin/uvicorn supernova.api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable supernova.service
sudo systemctl start supernova.service

# Check status
sudo systemctl status supernova.service
```

### 2. **Nginx Reverse Proxy Configuration**
```bash
# Install Nginx
sudo apt install nginx

# Create configuration
sudo nano /etc/nginx/sites-available/supernova
```

```nginx
server {
    listen 80;
    server_name your-domain.com api.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com api.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/supernova /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Monitoring and Logging

### 1. **Application Logging Configuration**
```python
# supernova/logging_config.py
import logging
import logging.handlers
import os
from .config import settings

def setup_logging():
    """Setup application logging"""
    
    # Create log directory
    log_dir = os.path.dirname(settings.LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=100*1024*1024,  # 100MB
                backupCount=5
            )
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Call in main application
setup_logging()
```

### 2. **Health Check Endpoint**
```python
# Add to supernova/api.py
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0",
            "database": "connected",
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
```

### 3. **Monitoring Setup with Prometheus (Optional)**
```bash
pip install prometheus-client
```

```python
# supernova/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
chat_requests_total = Counter('chat_requests_total', 'Total chat requests')
response_time_seconds = Histogram('response_time_seconds', 'Response time in seconds')
active_connections = Gauge('active_websocket_connections', 'Active WebSocket connections')

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()
```

---

## Backup and Recovery

### 1. **Database Backup Script**
```bash
#!/bin/bash
# backup_database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/supernova"
DB_NAME="supernova_db"
DB_USER="supernova_user"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -U $DB_USER -h localhost $DB_NAME | gzip > "$BACKUP_DIR/supernova_backup_$DATE.sql.gz"

# Keep only last 30 days of backups
find $BACKUP_DIR -name "supernova_backup_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: supernova_backup_$DATE.sql.gz"
```

### 2. **Automated Backup with Cron**
```bash
# Add to crontab
crontab -e

# Add line for daily backup at 2 AM
0 2 * * * /opt/supernova/scripts/backup_database.sh
```

---

## Testing in Production Environment

### 1. **Pre-Deployment Tests**
```python
# production_tests.py
import requests
import asyncio
import websockets
import json

async def test_api_endpoints():
    """Test API endpoints"""
    base_url = "https://your-domain.com"
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    print("✅ Health check passed")
    
    # Test authentication
    auth_data = {"username": "test", "password": "test123"}
    response = requests.post(f"{base_url}/auth/login", json=auth_data)
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("✅ Authentication passed")
        
        # Test authenticated endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{base_url}/auth/me", headers=headers)
        assert response.status_code == 200
        print("✅ Authenticated endpoint passed")

async def test_websocket():
    """Test WebSocket connection"""
    uri = "wss://your-domain.com/ws/chat"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "type": "message",
                "content": "Hello, test message"
            }))
            
            # Receive response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"✅ WebSocket test passed: {data}")
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())
    asyncio.run(test_websocket())
```

### 2. **Load Testing**
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 https://your-domain.com/health

# Test chat endpoint (with authentication)
ab -n 100 -c 5 -H "Authorization: Bearer your_token" -p chat_payload.json https://your-domain.com/chat
```

---

## Security Hardening

### 1. **Firewall Configuration**
```bash
# Configure UFW firewall
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw enable

# Block direct access to application port
sudo ufw deny 8000/tcp
```

### 2. **Application Security Headers**
```python
# Add to supernova/api.py
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### **1. Import Errors**
```bash
# Issue: ModuleNotFoundError
# Solution: Check Python path and virtual environment
python -c "import sys; print(sys.path)"
pip list | grep supernova
```

#### **2. Database Connection Issues**
```bash
# Issue: Cannot connect to database
# Check database service
sudo systemctl status postgresql

# Test connection manually
psql -U supernova_user -d supernova_db -h localhost
```

#### **3. LLM Provider API Errors**
```python
# Issue: API key not working
# Test API key directly
import openai
openai.api_key = "your_api_key"
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)
```

#### **4. WebSocket Connection Failures**
```bash
# Issue: WebSocket not connecting
# Check if port is open
netstat -tulpn | grep 8000

# Test WebSocket manually
wscat -c wss://your-domain.com/ws/chat
```

#### **5. High Memory Usage**
```bash
# Monitor memory usage
htop
ps aux | grep python

# Check application logs
tail -f /var/log/supernova/app.log
```

---

## Performance Optimization

### 1. **Database Optimization**
```sql
-- Create indexes for better performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(created_at);
```

### 2. **Application Optimization**
```python
# supernova/optimization.py
import asyncio
from functools import lru_cache

# Cache frequently used data
@lru_cache(maxsize=1000)
def get_user_preferences(user_id: int):
    """Cache user preferences"""
    # Implementation here
    pass

# Connection pooling for database
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

---

## Final Deployment Command

```bash
# Complete deployment script
#!/bin/bash

echo "🚀 Starting SuperNova Deployment..."

# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install python3 python3-pip nginx postgresql redis-server -y

# 3. Setup application
cd /opt/supernova
source venv/bin/activate
pip install -r requirements.txt

# 4. Setup database
python create_database.py

# 5. Start services
sudo systemctl start postgresql redis-server nginx
sudo systemctl start supernova.service

# 6. Test deployment
python production_tests.py

echo "✅ SuperNova deployment completed!"
echo "🌐 Access your application at: https://your-domain.com"
```

---

**🎉 Congratulations! Your SuperNova Conversational Agent is now ready for production deployment.**

For support and maintenance, monitor the application logs and health endpoints regularly. The system is designed to be highly available and scalable for your financial advisory needs.