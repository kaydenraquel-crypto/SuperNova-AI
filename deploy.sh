#!/bin/bash

# SuperNova AI Production Deployment Script
# This script handles the complete deployment process

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.prod.yml}
APP_NAME="supernova-ai"
BACKUP_DIR="/opt/backups/supernova"
LOG_FILE="/var/log/supernova-deploy.log"

# Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOG_FILE
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a $LOG_FILE
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check if environment file exists
    if [ ! -f .env.production ]; then
        warning ".env.production file not found. Creating from template..."
        cp .env.example .env.production || error "Failed to create .env.production"
    fi
    
    success "Prerequisites check completed"
}

# Create backup before deployment
create_backup() {
    info "Creating backup before deployment..."
    
    mkdir -p $BACKUP_DIR
    BACKUP_NAME="supernova-backup-$(date +%Y%m%d-%H%M%S)"
    
    # Backup database
    if docker ps | grep -q supernova-timescaledb; then
        docker exec supernova-timescaledb pg_dump -U supernova supernova_ts > "$BACKUP_DIR/$BACKUP_NAME-db.sql" || warning "Database backup failed"
    fi
    
    # Backup application data
    if docker volume ls | grep -q supernova-data; then
        docker run --rm -v supernova-data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/$BACKUP_NAME-data.tar.gz -C /data . || warning "Data backup failed"
    fi
    
    success "Backup created: $BACKUP_NAME"
}

# Build application
build_application() {
    info "Building SuperNova AI application..."
    
    # Build frontend
    info "Building frontend..."
    if [ -d "frontend" ]; then
        cd frontend
        npm ci --only=production || error "Frontend dependencies installation failed"
        npm run build || error "Frontend build failed"
        cd ..
        success "Frontend built successfully"
    else
        warning "Frontend directory not found, skipping frontend build"
    fi
    
    # Build Docker images
    info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache || error "Docker build failed"
    
    success "Application built successfully"
}

# Deploy application
deploy_application() {
    info "Deploying SuperNova AI..."
    
    # Load environment variables
    export $(cat .env.production | xargs)
    
    # Pull latest images (if using registry)
    info "Pulling latest images..."
    docker-compose -f $COMPOSE_FILE pull || warning "Image pull failed, continuing with local build"
    
    # Start services
    info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d || error "Failed to start services"
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    check_health
    
    success "Application deployed successfully"
}

# Health check
check_health() {
    info "Performing health checks..."
    
    # Check main application
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
            success "Application is healthy"
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        info "Health check attempt $RETRY_COUNT/$MAX_RETRIES..."
        sleep 10
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        error "Application health check failed after $MAX_RETRIES attempts"
    fi
    
    # Check database
    if docker exec supernova-timescaledb pg_isready -U supernova -d supernova_ts > /dev/null 2>&1; then
        success "Database is healthy"
    else
        warning "Database health check failed"
    fi
    
    # Check Redis
    if docker exec supernova-redis redis-cli ping | grep -q PONG; then
        success "Redis is healthy"
    else
        warning "Redis health check failed"
    fi
}

# Setup monitoring
setup_monitoring() {
    info "Setting up monitoring..."
    
    # Start monitoring services
    docker-compose -f $COMPOSE_FILE up -d prometheus grafana || warning "Monitoring setup failed"
    
    # Wait for Grafana to be ready
    sleep 20
    
    # Import dashboards (if available)
    if [ -d "docker/grafana/dashboards" ]; then
        info "Grafana dashboards will be auto-imported"
    fi
    
    success "Monitoring setup completed"
}

# Setup SSL certificates (Let's Encrypt)
setup_ssl() {
    info "Setting up SSL certificates..."
    
    if [ ! -d "docker/ssl" ]; then
        mkdir -p docker/ssl
        
        # Generate self-signed certificates for development
        if [ "$ENVIRONMENT" = "development" ]; then
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
                -keyout docker/ssl/key.pem \\
                -out docker/ssl/cert.pem \\
                -subj "/C=US/ST=CA/L=SF/O=SuperNova/CN=localhost" || warning "SSL certificate generation failed"
        else
            warning "SSL certificates not found. Please setup Let's Encrypt or provide certificates"
        fi
    fi
    
    success "SSL setup completed"
}

# Rollback function
rollback() {
    error "Deployment failed. Starting rollback..."
    
    # Stop current deployment
    docker-compose -f $COMPOSE_FILE down || warning "Failed to stop current deployment"
    
    # Restore from backup (if available)
    LATEST_BACKUP=$(ls -t $BACKUP_DIR/*-data.tar.gz 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        info "Restoring from backup: $LATEST_BACKUP"
        docker run --rm -v supernova-data:/data -v $BACKUP_DIR:/backup alpine tar xzf /backup/$(basename $LATEST_BACKUP) -C /data || warning "Backup restore failed"
    fi
    
    error "Rollback completed. Please check the issues and redeploy."
}

# Cleanup old resources
cleanup() {
    info "Cleaning up old resources..."
    
    # Remove unused Docker images
    docker image prune -f || warning "Image cleanup failed"
    
    # Remove unused volumes
    docker volume prune -f || warning "Volume cleanup failed"
    
    # Remove old backups (keep last 7 days)
    find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete || warning "Backup cleanup failed"
    find $BACKUP_DIR -name "*.sql" -mtime +7 -delete || warning "Database backup cleanup failed"
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting SuperNova AI deployment..."
    
    # Trap errors and rollback
    trap rollback ERR
    
    check_prerequisites
    create_backup
    setup_ssl
    build_application
    deploy_application
    setup_monitoring
    cleanup
    
    success "SuperNova AI deployment completed successfully!"
    
    info "Access points:"
    info "  - Main application: https://localhost"
    info "  - API documentation: https://localhost/docs"
    info "  - Monitoring (Grafana): http://localhost:3000"
    info "  - Monitoring (Prometheus): http://localhost:9090"
    info "  - Logs (Kibana): http://localhost:5601"
    
    info "To check status: docker-compose -f $COMPOSE_FILE ps"
    info "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
}

# Command line options
case "$1" in
    "build")
        build_application
        ;;
    "deploy")
        deploy_application
        ;;
    "health")
        check_health
        ;;
    "backup")
        create_backup
        ;;
    "cleanup")
        cleanup
        ;;
    "rollback")
        rollback
        ;;
    *)
        main
        ;;
esac