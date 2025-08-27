# SuperNova AI Installation Guide

Welcome to SuperNova AI! This guide will walk you through the complete installation process for both end users and developers.

## Quick Start for End Users

### System Requirements

**Minimum Requirements:**
- **Operating System**: Windows 10+, macOS 10.15+, or Linux Ubuntu 18.04+
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 5GB available disk space
- **Internet**: Stable broadband connection
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+

**Recommended Requirements:**
- **Memory**: 16GB RAM or higher
- **Storage**: 20GB available disk space (for data caching)
- **Processor**: Intel i5 8th gen / AMD Ryzen 5 3600 or equivalent

### Pre-Installation Checklist

Before installing SuperNova AI, ensure you have:

- [ ] Administrator privileges on your computer
- [ ] Stable internet connection
- [ ] Updated web browser
- [ ] Antivirus software configured to allow new installations
- [ ] At least 5GB free disk space

## Installation Options

### Option 1: Docker Installation (Recommended)

The easiest way to install SuperNova AI is using Docker, which handles all dependencies automatically.

#### Step 1: Install Docker

**Windows:**
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Run the installer and follow the setup wizard
3. Restart your computer when prompted
4. Verify installation: Open Command Prompt and run `docker --version`

**macOS:**
1. Download Docker Desktop for Mac
2. Drag Docker to Applications folder
3. Launch Docker from Applications
4. Verify installation: Open Terminal and run `docker --version`

**Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install Docker
sudo apt install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER

# Verify installation
docker --version
```

#### Step 2: Download SuperNova AI

1. Create a directory for SuperNova AI:
   ```bash
   mkdir supernova-ai
   cd supernova-ai
   ```

2. Download the SuperNova AI Docker configuration:
   ```bash
   # Download docker-compose.yml
   curl -O https://raw.githubusercontent.com/supernova-ai/supernova/main/docker-compose.yml
   
   # Download environment template
   curl -O https://raw.githubusercontent.com/supernova-ai/supernova/main/.env.example
   ```

#### Step 3: Configure Environment

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your preferred settings:
   ```bash
   # Basic Configuration
   SUPERNOVA_HOST=0.0.0.0
   SUPERNOVA_PORT=8081
   SUPERNOVA_DEBUG=false
   
   # Database Configuration
   DATABASE_URL=sqlite:///./supernova.db
   
   # LLM Configuration (Optional - for AI features)
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Security Configuration
   SECRET_KEY=your_secret_key_here_min_32_chars
   JWT_SECRET_KEY=your_jwt_secret_here
   
   # WebSocket Configuration
   ENABLE_WEBSOCKETS=true
   WEBSOCKET_PORT=8082
   ```

#### Step 4: Launch SuperNova AI

1. Start the application:
   ```bash
   docker-compose up -d
   ```

2. Verify the installation:
   ```bash
   docker-compose ps
   ```

   You should see services running:
   - `supernova-backend` (API server)
   - `supernova-frontend` (Web interface)
   - `supernova-db` (Database)

3. Access SuperNova AI:
   - Web Interface: http://localhost:3000
   - API Documentation: http://localhost:8081/docs

### Option 2: Manual Installation

For advanced users who prefer manual installation.

#### Prerequisites

**Python Requirements:**
- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment support

**Node.js Requirements (for frontend):**
- Node.js 16.0 or higher
- npm 8.0 or higher

#### Step 1: Install Python Dependencies

1. Download the SuperNova AI source code:
   ```bash
   git clone https://github.com/supernova-ai/supernova.git
   cd supernova
   ```

2. Create and activate virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Step 2: Install Frontend Dependencies

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

#### Step 3: Database Setup

1. Initialize the database:
   ```bash
   python -m supernova.db init
   ```

2. Create sample data (optional):
   ```bash
   python -m supernova.db seed
   ```

#### Step 4: Configuration

1. Create environment configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings (same as Docker installation above)

#### Step 5: Launch Application

1. Start the backend API:
   ```bash
   # From root directory with venv activated
   uvicorn supernova.api:app --reload --host 0.0.0.0 --port 8081
   ```

2. Start the frontend (in a new terminal):
   ```bash
   cd frontend
   npm start
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8081/docs

## Post-Installation Setup

### Initial Configuration

1. **Access the Web Interface**
   - Open your web browser
   - Navigate to http://localhost:3000
   - You should see the SuperNova AI login screen

2. **Create Your First Account**
   - Click "Sign Up" or "Create Account"
   - Fill in your basic information:
     - Full Name
     - Email Address
     - Password (minimum 8 characters)
   - Click "Create Account"

3. **Complete Your Profile**
   - Answer the risk assessment questions
   - Set your investment objectives
   - Define any constraints or preferences
   - Review and save your profile

### LLM Provider Setup (Optional)

To enable AI advisory features, you'll need API keys from LLM providers:

#### OpenAI Setup
1. Visit [platform.openai.com](https://platform.openai.com)
2. Create an account and navigate to API keys
3. Generate a new API key
4. Add to your `.env` file: `OPENAI_API_KEY=your_key_here`

#### Anthropic Claude Setup
1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Create an account and navigate to API keys
3. Generate a new API key
4. Add to your `.env` file: `ANTHROPIC_API_KEY=your_key_here`

### Verification Steps

After installation, verify everything is working:

1. **Backend Health Check**
   ```bash
   curl http://localhost:8081/health
   ```
   Should return: `{"status": "healthy"}`

2. **Frontend Accessibility**
   - Navigate to http://localhost:3000
   - Login page should load completely
   - No console errors in browser developer tools

3. **Database Connectivity**
   - Try creating a test user account
   - Verify data persists after browser refresh

4. **WebSocket Connection**
   - Enable real-time features in settings
   - Verify live updates work (if applicable)

## Troubleshooting Common Issues

### Docker Installation Issues

**Issue: Docker service not running**
```bash
# Windows (PowerShell as Administrator)
Start-Service docker

# Linux
sudo systemctl start docker
```

**Issue: Port conflicts**
```bash
# Check which process is using port 8081
netstat -tulpn | grep :8081

# Kill the process (replace PID with actual process ID)
kill -9 PID
```

### Manual Installation Issues

**Issue: Python version compatibility**
```bash
# Check Python version
python --version

# If using Python 3.9+, ensure pip is updated
pip install --upgrade pip
```

**Issue: Permission errors (Linux/macOS)**
```bash
# Fix permissions
chmod +x scripts/setup.sh
sudo chown -R $USER:$USER .
```

**Issue: Node.js build failures**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### Database Issues

**Issue: Database connection errors**
1. Verify database file permissions
2. Check DATABASE_URL in .env file
3. Reinitialize database:
   ```bash
   rm supernova.db
   python -m supernova.db init
   ```

**Issue: Migration errors**
```bash
# Reset database (WARNING: destroys all data)
python -m supernova.db reset
python -m supernova.db init
python -m supernova.db seed
```

### Network and Security Issues

**Issue: Firewall blocking connections**
- Windows: Add exceptions for ports 3000 and 8081
- macOS: System Preferences > Security & Privacy > Firewall Options
- Linux: Configure iptables or ufw to allow ports

**Issue: SSL/HTTPS errors**
- For development, HTTP is acceptable
- For production, configure proper SSL certificates
- Update browser security settings if needed

## Performance Optimization

### System Performance

1. **Memory Optimization**
   - Allocate at least 8GB RAM to Docker (if using Docker Desktop)
   - Close unnecessary applications during use
   - Monitor memory usage in Task Manager/Activity Monitor

2. **Storage Optimization**
   - Use SSD storage for better performance
   - Ensure adequate free space (20% minimum)
   - Regular cleanup of temporary files

3. **Network Optimization**
   - Use wired connection when possible
   - Ensure stable internet (minimum 10 Mbps)
   - Configure DNS settings for optimal speed

### Application Performance

1. **Database Performance**
   - Regular database maintenance
   - Monitor query performance
   - Consider upgrading to PostgreSQL for production

2. **API Performance**
   - Enable caching in production
   - Monitor API response times
   - Configure rate limiting appropriately

## Security Considerations

### Development Environment

- Change default passwords immediately
- Use strong, unique passwords
- Enable two-factor authentication when available
- Keep software updated regularly

### Production Environment

- Use HTTPS encryption
- Configure proper firewall rules
- Regular security audits
- Backup data regularly
- Monitor for suspicious activity

## Next Steps

After successful installation:

1. **Read the User Manual**: [User Manual](user-manual.md)
2. **Complete the Tutorial**: [Getting Started Tutorial](../tutorials/basic-tutorial.md)
3. **Explore Features**: [Feature Overview](platform-overview.md)
4. **Join the Community**: [Community Resources](../business/community-resources.md)

## Support and Help

If you encounter issues during installation:

1. **Check Documentation**: Search our comprehensive docs
2. **Community Forum**: Post questions to our community
3. **GitHub Issues**: Report bugs on our GitHub repository
4. **Professional Support**: Contact support for enterprise assistance

**Support Channels:**
- Documentation: https://docs.supernova-ai.com
- Community Forum: https://community.supernova-ai.com
- GitHub Issues: https://github.com/supernova-ai/supernova/issues
- Email Support: support@supernova-ai.com

---

**Next: [Getting Started Guide](getting-started.md)**

Last Updated: August 26, 2025  
Version: 1.0.0