# Production Deployment Guide

This comprehensive guide covers deploying SuperNova AI to production environments, including cloud platforms, containerization, monitoring, and best practices for enterprise deployment.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Platform Deployment](#cloud-platform-deployment)
5. [Database Setup](#database-setup)
6. [Load Balancing and Scaling](#load-balancing-and-scaling)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [Environment Configuration](#environment-configuration)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Monitoring and Logging](#monitoring-and-logging)
11. [Backup and Recovery](#backup-and-recovery)
12. [Security Hardening](#security-hardening)

## Deployment Overview

### Architecture Components

SuperNova AI production deployment consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
│                 (nginx/AWS ALB)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼──────────┐      ┌───────▼──────────┐
│   Frontend App    │      │   API Servers    │
│   (React/nginx)   │      │   (FastAPI)      │
│   - Static Files  │      │   - Multiple     │
│   - SPA Router    │      │     Instances    │
└───────────────────┘      │   - Auto-scaling │
                          └───────┬──────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
┌────────▼──────────┐      ┌───────▼──────────┐    ┌──────▼────┐
│    Database       │      │     Cache         │    │  Storage  │
│  (PostgreSQL/     │      │    (Redis)        │    │   (S3/    │
│   TimescaleDB)    │      │  - Sessions       │    │   GCS)    │
│  - Primary DB     │      │  - API Cache      │    │ - Backups │
│  - Read Replicas  │      │  - WebSockets     │    │ - Logs    │
└───────────────────┘      └───────────────────┘    └───────────┘
```

### Deployment Strategies

1. **Blue-Green Deployment**: Zero-downtime deployments with instant rollback
2. **Rolling Deployment**: Gradual replacement of instances
3. **Canary Deployment**: Gradual traffic shift to new version
4. **Multi-Region**: High availability across geographic regions

## Infrastructure Requirements

### Minimum Production Requirements

#### Backend Infrastructure
- **CPU**: 4 vCPUs per API server instance
- **Memory**: 16GB RAM per instance (32GB recommended)
- **Storage**: 100GB SSD for application + database storage
- **Network**: 1Gbps network connection
- **Load Balancer**: Minimum 2 API server instances

#### Database Infrastructure
- **PostgreSQL**: Dedicated server with 8 vCPUs, 32GB RAM
- **TimescaleDB**: For time-series data, 4 vCPUs, 16GB RAM
- **Redis**: 2 vCPUs, 8GB RAM for caching
- **Storage**: High-performance SSD with IOPS optimization

#### Recommended Production Setup
- **API Servers**: 3-5 instances behind load balancer
- **Database**: Primary + 2 read replicas
- **Cache**: Redis cluster with failover
- **Monitoring**: Dedicated monitoring stack
- **Backup**: Automated backup system

### Cloud Platform Sizing

#### AWS EC2 Instance Types
```yaml
# API Servers
api_servers:
  instance_type: "c5.xlarge"  # 4 vCPUs, 8GB RAM
  min_instances: 2
  max_instances: 10
  
# Database
database:
  rds_instance: "db.r5.2xlarge"  # 8 vCPUs, 64GB RAM
  storage: "500GB gp3"
  backup_retention: 30

# Cache
cache:
  elasticache: "cache.r6g.large"  # 2 vCPUs, 13GB RAM
  node_type: "redis"
  
# Load Balancer  
load_balancer:
  type: "Application Load Balancer"
  scheme: "internet-facing"
```

#### Google Cloud Platform
```yaml
# Compute Engine Instances
api_servers:
  machine_type: "e2-standard-4"  # 4 vCPUs, 16GB RAM
  zone: "us-central1-a"
  
# Cloud SQL
database:
  tier: "db-custom-8-30720"  # 8 vCPUs, 30GB RAM
  storage_size: "500GB"
  storage_type: "SSD"

# Memorystore Redis
cache:
  tier: "standard"
  memory_size_gb: 16
```

#### Microsoft Azure
```yaml
# Virtual Machines
api_servers:
  vm_size: "Standard_D4s_v3"  # 4 vCPUs, 16GB RAM
  
# Azure Database for PostgreSQL
database:
  sku_name: "GP_Gen5_8"  # 8 vCPUs, General Purpose
  storage_mb: 512000
  
# Azure Cache for Redis
cache:
  sku_name: "Standard"
  family: "C"
  capacity: 6
```

## Docker Deployment

### Production Docker Configuration

#### Multi-Stage Dockerfile

**Backend Dockerfile:**
```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash supernova
USER supernova
WORKDIR /home/supernova/app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/supernova/.local
ENV PATH=/home/supernova/.local/bin:$PATH

# Copy application code
COPY --chown=supernova:supernova . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/health || exit 1

EXPOSE 8081
CMD ["uvicorn", "supernova.api:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "4"]
```

**Frontend Dockerfile:**
```dockerfile
# Build stage
FROM node:16-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose for Production

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/ssl/private:ro
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - backend
    networks:
      - supernova-network

  # Backend API
  backend:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://supernova:${DB_PASSWORD}@postgres:5432/supernova
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - supernova-network
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  # Database
  postgres:
    image: timescale/timescaledb:latest-pg13
    restart: unless-stopped
    environment:
      - POSTGRES_DB=supernova
      - POSTGRES_USER=supernova
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backup:/backup
    networks:
      - supernova-network
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G

  # Cache
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - supernova-network

  # WebSocket Server
  websocket:
    build:
      context: .
      dockerfile: Dockerfile.websocket
    restart: unless-stopped
    ports:
      - "8082:8082"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    networks:
      - supernova-network

networks:
  supernova-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
```

### Container Orchestration

#### Docker Swarm Deployment

**Initialize Swarm:**
```bash
# Initialize swarm
docker swarm init

# Add worker nodes
docker swarm join --token <worker-token> <manager-ip>:2377

# Deploy stack
docker stack deploy -c docker-compose.prod.yml supernova
```

**Stack Configuration:**
```yaml
version: '3.8'

services:
  backend:
    image: supernova/api:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
    networks:
      - supernova-overlay

networks:
  supernova-overlay:
    driver: overlay
    attachable: true
```

#### Kubernetes Deployment

**Namespace:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: supernova
```

**Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supernova-api
  namespace: supernova
spec:
  replicas: 3
  selector:
    matchLabels:
      app: supernova-api
  template:
    metadata:
      labels:
        app: supernova-api
    spec:
      containers:
      - name: supernova-api
        image: supernova/api:latest
        ports:
        - containerPort: 8081
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: supernova-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: supernova-secrets
              key: secret-key
        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
          requests:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: supernova-api-service
  namespace: supernova
spec:
  selector:
    app: supernova-api
  ports:
  - port: 80
    targetPort: 8081
  type: LoadBalancer
```

## Cloud Platform Deployment

### AWS Deployment

#### ECS with Fargate

**Task Definition:**
```json
{
  "family": "supernova-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "supernova-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/supernova:latest",
      "portMappings": [
        {
          "containerPort": 8081,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/supernova"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:supernova/secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/supernova-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Service Configuration:**
```json
{
  "serviceName": "supernova-api-service",
  "cluster": "supernova-cluster",
  "taskDefinition": "supernova-api:1",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-12345", "subnet-67890"],
      "securityGroups": ["sg-abcdef"],
      "assignPublicIp": "ENABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/supernova-tg",
      "containerName": "supernova-api",
      "containerPort": 8081
    }
  ]
}
```

#### RDS Configuration

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier supernova-prod \
  --db-instance-class db.r5.2xlarge \
  --engine postgres \
  --engine-version 13.7 \
  --master-username supernovauser \
  --master-user-password ${DB_PASSWORD} \
  --allocated-storage 500 \
  --storage-type gp2 \
  --storage-encrypted \
  --backup-retention-period 30 \
  --multi-az \
  --publicly-accessible false \
  --vpc-security-group-ids sg-12345678
```

### Google Cloud Platform Deployment

#### Cloud Run Configuration

**service.yaml:**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: supernova-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/project-id/supernova:latest
        ports:
        - containerPort: 8081
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: supernova-secrets
              key: database-url
        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
```

#### Cloud SQL Setup

```bash
# Create Cloud SQL instance
gcloud sql instances create supernova-prod \
  --database-version=POSTGRES_13 \
  --tier=db-custom-8-30720 \
  --storage-size=500GB \
  --storage-type=SSD \
  --backup-start-time=02:00 \
  --backup-location=us-central1 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=06 \
  --availability-type=REGIONAL
```

### Azure Deployment

#### Container Instances

**deployment-template.json:**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "supernova-api"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2019-12-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "supernova-api",
            "properties": {
              "image": "supernovarepo.azurecr.io/api:latest",
              "ports": [
                {
                  "port": 8081,
                  "protocol": "TCP"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "Always",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8081,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ]
}
```

## Database Setup

### PostgreSQL Production Configuration

#### Master Database Setup

**postgresql.conf optimizations:**
```ini
# Memory Settings
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
work_mem = 256MB
maintenance_work_mem = 2GB

# Checkpoint Settings
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min
max_wal_size = 4GB
min_wal_size = 1GB

# Connection Settings
max_connections = 200
shared_preload_libraries = 'pg_stat_statements,timescaledb'

# Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_min_duration_statement = 1000
```

#### Read Replicas Configuration

```bash
# Create read replica (AWS RDS)
aws rds create-db-instance-read-replica \
  --db-instance-identifier supernova-replica-1 \
  --source-db-instance-identifier supernova-prod \
  --db-instance-class db.r5.large \
  --availability-zone us-west-2b

# Create read replica (GCP Cloud SQL)
gcloud sql instances create supernova-replica \
  --master-instance-name=supernova-prod \
  --tier=db-custom-4-15360 \
  --replica-type=READ \
  --zone=us-central1-b
```

### TimescaleDB Configuration

#### Hypertables Setup

```sql
-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create sentiment data hypertable
CREATE TABLE sentiment_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    overall_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    social_momentum FLOAT,
    news_sentiment FLOAT,
    source_counts JSONB,
    PRIMARY KEY (timestamp, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('sentiment_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX ON sentiment_data (symbol, timestamp DESC);
CREATE INDEX ON sentiment_data (timestamp DESC, overall_score);
CREATE INDEX ON sentiment_data USING GIN (source_counts);

-- Set up data retention policy (keep 1 year)
SELECT add_retention_policy('sentiment_data', INTERVAL '1 year');

-- Create continuous aggregates for better query performance
CREATE MATERIALIZED VIEW sentiment_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', timestamp) AS hour,
       symbol,
       AVG(overall_score) AS avg_sentiment,
       COUNT(*) AS data_points
FROM sentiment_data
GROUP BY hour, symbol;

-- Enable automatic refresh
SELECT add_continuous_aggregate_policy('sentiment_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

### Database Connection Pooling

#### PgBouncer Configuration

**pgbouncer.ini:**
```ini
[databases]
supernova = host=postgres-master port=5432 dbname=supernova
supernova_readonly = host=postgres-replica port=5432 dbname=supernova

[pgbouncer]
pool_mode = transaction
listen_port = 6432
listen_addr = 0.0.0.0
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid

# Connection pool settings
max_client_conn = 1000
default_pool_size = 20
max_db_connections = 50
pool_mode = transaction
server_reset_query = DISCARD ALL
ignore_startup_parameters = extra_float_digits

# Timeouts
server_connect_timeout = 15
server_idle_timeout = 600
server_lifetime = 3600
client_idle_timeout = 0
```

## Load Balancing and Scaling

### Nginx Load Balancer Configuration

**nginx.conf:**
```nginx
upstream supernova_backend {
    least_conn;
    server api1.supernova.local:8081 weight=1 max_fails=3 fail_timeout=30s;
    server api2.supernova.local:8081 weight=1 max_fails=3 fail_timeout=30s;
    server api3.supernova.local:8081 weight=1 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream supernova_websocket {
    ip_hash;  # Ensure session persistence
    server ws1.supernova.local:8082;
    server ws2.supernova.local:8082;
    server ws3.supernova.local:8082;
}

server {
    listen 80;
    listen [::]:80;
    server_name api.supernova-ai.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.supernova-ai.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/supernova.crt;
    ssl_certificate_key /etc/ssl/private/supernova.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # API routes
    location /api {
        proxy_pass http://supernova_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Keep connections alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # WebSocket routes
    location /ws {
        proxy_pass http://supernova_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # WebSocket timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

server {
    # Apply rate limiting
    location /api {
        limit_req zone=api burst=20 nodelay;
        # ... rest of configuration
    }
    
    location /auth {
        limit_req zone=auth burst=5 nodelay;
        # ... rest of configuration
    }
}
```

### Auto-scaling Configuration

#### AWS Auto Scaling Group

**autoscaling.yaml:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: supernova-api-template
      LaunchTemplateData:
        ImageId: ami-0abcdef1234567890  # SuperNova API AMI
        InstanceType: c5.xlarge
        SecurityGroupIds: 
          - !Ref InstanceSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash
            docker run -d --name supernova-api \
              -p 8081:8081 \
              -e DATABASE_URL="${DatabaseURL}" \
              supernova/api:latest
  
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: supernova-api-asg
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 2
      MaxSize: 10
      DesiredCapacity: 3
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300
  
  ScalingPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      PolicyType: TargetTrackingScaling
      AutoScalingGroupName: !Ref AutoScalingGroup
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 70.0
```

#### Kubernetes Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: supernova-api-hpa
  namespace: supernova
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: supernova-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

## SSL/TLS Configuration

### Certificate Management

#### Let's Encrypt with Certbot

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.supernova-ai.com -d www.supernova-ai.com

# Automatic renewal setup
sudo crontab -e
# Add line: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### AWS Certificate Manager

```bash
# Request certificate via CLI
aws acm request-certificate \
  --domain-name supernova-ai.com \
  --subject-alternative-names www.supernova-ai.com api.supernova-ai.com \
  --validation-method DNS

# Describe certificate for validation
aws acm describe-certificate --certificate-arn arn:aws:acm:region:account:certificate/cert-id
```

### SSL Configuration Best Practices

**nginx SSL configuration:**
```nginx
# Modern SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# HSTS
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# Session resumption
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
```

## Environment Configuration

### Production Environment Variables

**production.env:**
```bash
# Application Settings
NODE_ENV=production
DEBUG=false
SUPERNOVA_HOST=0.0.0.0
SUPERNOVA_PORT=8081
WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://user:pass@postgres-cluster.internal:5432/supernova
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30

# Redis Configuration
REDIS_URL=redis://redis-cluster.internal:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_MAX_CONNECTIONS=100

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=1
REFRESH_TOKEN_EXPIRATION_DAYS=30

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_BURST_SIZE=100

# External APIs
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
NOVASIGNAL_API_KEY=${NOVASIGNAL_API_KEY}

# Monitoring and Logging
LOG_LEVEL=INFO
SENTRY_DSN=${SENTRY_DSN}
DATADOG_API_KEY=${DATADOG_API_KEY}

# Feature Flags
ENABLE_WEBSOCKETS=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_ADVANCED_FEATURES=true

# Performance
VECTORBT_ENABLED=true
VECTORBT_DEFAULT_FEES=0.001
BACKTEST_MAX_BARS=10000
CACHE_TTL_SECONDS=3600
```

### Secrets Management

#### AWS Secrets Manager

```bash
# Store secrets
aws secretsmanager create-secret \
  --name "supernova/production" \
  --description "SuperNova production secrets" \
  --secret-string '{
    "database_password": "secure_password_here",
    "secret_key": "32_char_secret_key_here",
    "openai_api_key": "sk-openai-key-here"
  }'

# Retrieve secrets in application
aws secretsmanager get-secret-value \
  --secret-id "supernova/production" \
  --query SecretString --output text
```

#### Google Secret Manager

```bash
# Create secrets
echo -n "secure_database_password" | gcloud secrets create database-password --data-file=-
echo -n "32_character_secret_key_here" | gcloud secrets create secret-key --data-file=-

# Access secrets
gcloud secrets versions access latest --secret="database-password"
```

## CI/CD Pipeline

### GitHub Actions Workflow

**.github/workflows/deploy.yml:**
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          
      - name: Run tests
        run: |
          pytest --cov=supernova --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.prod
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        uses: azure/k8s-deploy@v1
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

### Blue-Green Deployment Script

**deploy.sh:**
```bash
#!/bin/bash
set -e

# Configuration
CLUSTER_NAME="supernova-prod"
SERVICE_NAME="supernova-api"
NEW_TAG=$1

if [ -z "$NEW_TAG" ]; then
    echo "Usage: $0 <new-image-tag>"
    exit 1
fi

echo "Starting blue-green deployment..."

# Create new task definition
TASK_DEF=$(aws ecs describe-task-definition --task-definition $SERVICE_NAME --query 'taskDefinition')
NEW_TASK_DEF=$(echo $TASK_DEF | jq --arg IMAGE "$SERVICE_NAME:$NEW_TAG" '.containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .placementConstraints, .compatibilities, .registeredAt, .registeredBy)')

# Register new task definition
NEW_REVISION=$(aws ecs register-task-definition --cli-input-json "$NEW_TASK_DEF" --query 'taskDefinition.revision')

echo "Created new task definition revision: $NEW_REVISION"

# Update service with new task definition
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --task-definition $SERVICE_NAME:$NEW_REVISION

echo "Service updated. Waiting for deployment to complete..."

# Wait for deployment to complete
aws ecs wait services-stable \
  --cluster $CLUSTER_NAME \
  --services $SERVICE_NAME

echo "Deployment completed successfully!"

# Health check
ENDPOINT="https://api.supernova-ai.com/health"
if curl -f -s $ENDPOINT > /dev/null; then
    echo "Health check passed!"
else
    echo "Health check failed! Rolling back..."
    # Implement rollback logic here
    exit 1
fi
```

## Monitoring and Logging

### Application Monitoring

#### Prometheus Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "supernova_rules.yml"

scrape_configs:
  - job_name: 'supernova-api'
    static_configs:
      - targets: ['api1:8081', 'api2:8081', 'api3:8081']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Grafana Dashboard

**supernova-dashboard.json:**
```json
{
  "dashboard": {
    "title": "SuperNova AI Production Monitoring",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Centralized Logging

#### ELK Stack Configuration

**logstash.conf:**
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][app] == "supernova-api" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger} %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "supernova-logs-%{+YYYY.MM.dd}"
  }
}
```

**filebeat.yml:**
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/supernova/*.log
  fields:
    app: supernova-api
  fields_under_root: true

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

## Backup and Recovery

### Database Backup Strategy

#### Automated PostgreSQL Backups

**backup-script.sh:**
```bash
#!/bin/bash

# Configuration
DB_HOST="postgres-prod.internal"
DB_NAME="supernova"
DB_USER="backup_user"
BACKUP_DIR="/backups/postgresql"
S3_BUCKET="supernova-backups"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
BACKUP_FILE="supernova_$(date +%Y%m%d_%H%M%S).sql.gz"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"

# Perform backup
echo "Starting backup at $(date)"
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME --no-password | gzip > $BACKUP_PATH

# Verify backup
if [ -f $BACKUP_PATH ]; then
    echo "Backup created successfully: $BACKUP_PATH"
    BACKUP_SIZE=$(du -h $BACKUP_PATH | cut -f1)
    echo "Backup size: $BACKUP_SIZE"
else
    echo "Backup failed!"
    exit 1
fi

# Upload to S3
aws s3 cp $BACKUP_PATH s3://$S3_BUCKET/postgresql/
if [ $? -eq 0 ]; then
    echo "Backup uploaded to S3 successfully"
else
    echo "S3 upload failed!"
fi

# Clean up old backups locally
find $BACKUP_DIR -name "supernova_*.sql.gz" -mtime +7 -delete

# Clean up old backups in S3
aws s3 ls s3://$S3_BUCKET/postgresql/ | while read -r line; do
    file_date=$(echo $line | awk '{print $1" "$2}')
    file_name=$(echo $line | awk '{print $4}')
    file_epoch=$(date -d "$file_date" +%s)
    current_epoch=$(date +%s)
    
    if [ $((current_epoch - file_epoch)) -gt $((RETENTION_DAYS * 86400)) ]; then
        echo "Deleting old backup: $file_name"
        aws s3 rm s3://$S3_BUCKET/postgresql/$file_name
    fi
done

echo "Backup completed at $(date)"
```

#### Point-in-Time Recovery Setup

**postgresql.conf:**
```ini
# WAL settings for PITR
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://supernova-wal-backups/%f'
archive_timeout = 300
max_wal_senders = 3
```

### Application Data Backup

**app-backup.sh:**
```bash
#!/bin/bash

# Backup user-uploaded files
rsync -av --delete /app/uploads/ /backups/uploads/

# Backup configuration files
tar -czf /backups/config/config-$(date +%Y%m%d).tar.gz /app/config/

# Backup logs (last 7 days)
find /var/log/supernova -name "*.log" -mtime -7 | xargs tar -czf /backups/logs/logs-$(date +%Y%m%d).tar.gz

# Upload to cloud storage
aws s3 sync /backups/ s3://supernova-backups/application/
```

## Security Hardening

### Network Security

#### Firewall Configuration

**ufw rules:**
```bash
# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access
sudo ufw allow ssh

# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Application ports (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 8081
sudo ufw allow from 10.0.0.0/8 to any port 5432
sudo ufw allow from 10.0.0.0/8 to any port 6379

# Enable firewall
sudo ufw enable
```

#### Security Group Rules (AWS)

```json
{
  "SecurityGroupRules": [
    {
      "IpPermissions": [
        {
          "IpProtocol": "tcp",
          "FromPort": 443,
          "ToPort": 443,
          "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
        },
        {
          "IpProtocol": "tcp",
          "FromPort": 8081,
          "ToPort": 8081,
          "UserIdGroupPairs": [{"GroupId": "sg-loadbalancer"}]
        }
      ]
    }
  ]
}
```

### Application Security

#### Security Headers

```python
# FastAPI security middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.supernova-ai.com", "*.supernova-ai.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://supernova-ai.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### Vulnerability Scanning

#### Regular Security Scans

**security-scan.sh:**
```bash
#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade -y

# Scan for vulnerabilities with Lynis
sudo lynis audit system

# Check for outdated packages
apt list --upgradable

# Scan Docker images
docker scan supernova/api:latest

# Check SSL configuration
testssl.sh https://api.supernova-ai.com

# Generate security report
echo "Security scan completed on $(date)" > /var/log/security-scan.log
```

---

## Production Checklist

### Pre-Deployment Checklist

- [ ] All tests passing in CI/CD pipeline
- [ ] Security scan completed with no critical issues
- [ ] Database migrations tested and ready
- [ ] SSL certificates installed and verified
- [ ] Load balancer health checks configured
- [ ] Monitoring and alerting set up
- [ ] Backup systems tested and verified
- [ ] Disaster recovery plan documented
- [ ] Performance benchmarks established
- [ ] Security configurations hardened

### Post-Deployment Verification

- [ ] Application accessible via load balancer
- [ ] Database connections working
- [ ] Authentication system functioning
- [ ] WebSocket connections established
- [ ] API endpoints responding correctly
- [ ] Monitoring systems collecting data
- [ ] Log aggregation working
- [ ] Backup jobs scheduled and running
- [ ] SSL/TLS certificates valid
- [ ] Performance within acceptable limits

---

**Related Documentation:**
- [Docker Guide](docker-guide.md)
- [Security Guide](../security/security-overview.md)
- [Monitoring Setup](../operations/monitoring.md)
- [Backup and Recovery](../operations/backup-recovery.md)

Last Updated: August 26, 2025  
Version: 1.0.0