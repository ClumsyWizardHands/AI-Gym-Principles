# AI Principles Gym - Production Deployment Guide

## Table of Contents
1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Deployment Steps](#deployment-steps)
3. [Scaling Considerations](#scaling-considerations)
4. [Monitoring Setup](#monitoring-setup)
5. [Security Hardening](#security-hardening)
6. [Operational Procedures](#operational-procedures)

## Infrastructure Requirements

### Core Services Architecture
```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   Load Balancer │────▶│  API Servers │────▶│  PostgreSQL  │
│   (nginx/ALB)   │     │  (2+ nodes)  │     │  (Primary)   │
└─────────────────┘     └──────────────┘     └──────────────┘
                               │                      │
                               ▼                      ▼
                        ┌──────────────┐     ┌──────────────┐
                        │ Redis Cluster│     │  PostgreSQL  │
                        │ (3+ nodes)   │     │  (Replica)   │
                        └──────────────┘     └──────────────┘
```

### Minimum Production Requirements

#### API Servers (2+ nodes)
- **CPU**: 4 vCPUs per node (Intel/AMD x86_64)
- **Memory**: 8GB RAM per node
- **Storage**: 50GB SSD for logs and temporary data
- **Network**: 1Gbps dedicated bandwidth
- **OS**: Ubuntu 22.04 LTS or RHEL 8+

#### PostgreSQL Database
- **Primary Node**:
  - CPU: 8 vCPUs
  - Memory: 32GB RAM
  - Storage: 500GB NVMe SSD (10,000+ IOPS)
  - Network: 10Gbps for replication
- **Replica Nodes** (1+ for HA):
  - Same specs as primary
  - Streaming replication with < 1s lag

#### Redis Cluster
- **Nodes**: Minimum 3 nodes for HA
- **Per Node**:
  - CPU: 2 vCPUs
  - Memory: 8GB RAM
  - Storage: 20GB SSD
- **Configuration**: Redis Sentinel for automatic failover

#### Load Balancer
- **nginx** (self-managed) or **AWS ALB/GCP LB** (cloud-managed)
- SSL/TLS termination
- Health check endpoints
- Request routing and rate limiting

#### Additional Services
- **Monitoring Stack**: Prometheus (2GB RAM) + Grafana (1GB RAM)
- **Log Aggregation**: ELK Stack (8GB RAM minimum)
- **Backup Storage**: S3-compatible storage for database backups

## Deployment Steps

### 1. Infrastructure Setup with Terraform

Create `terraform/main.tf`:

```hcl
# Provider Configuration
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

variable "db_password" {
  sensitive = true
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "principles-gym-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  
  tags = {
    Environment = var.environment
  }
}

# Security Groups
resource "aws_security_group" "api_servers" {
  name_prefix = "principles-gym-api"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres_primary" {
  identifier = "principles-gym-db"
  
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 500
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "principles_gym"
  username = "gymadmin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  tags = {
    Environment = var.environment
  }
}

# Read Replica
resource "aws_db_instance" "postgres_replica" {
  identifier = "principles-gym-db-replica"
  
  replicate_source_db = aws_db_instance.postgres_primary.identifier
  instance_class      = "db.r6g.xlarge"
  
  publicly_accessible = false
  auto_minor_version_upgrade = false
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "principles-gym-redis"
  description          = "Redis cluster for session and cache"
  
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_clusters   = 3
  port                 = 6379
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Environment = var.environment
  }
}

# ECS Cluster for API Servers
resource "aws_ecs_cluster" "main" {
  name = "principles-gym-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "principles-gym-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "4096"
  memory                   = "8192"
  
  container_definitions = jsonencode([{
    name  = "api"
    image = "your-registry/principles-gym:latest"
    
    environment = [
      {
        name  = "DATABASE_URL"
        value = "postgresql+asyncpg://gymadmin:${var.db_password}@${aws_db_instance.postgres_primary.endpoint}/principles_gym"
      },
      {
        name  = "CACHE_REDIS_URL"
        value = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379/0"
      }
    ]
    
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

# ECS Service with Auto Scaling
resource "aws_ecs_service" "api" {
  name            = "principles-gym-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  
  launch_type = "FARGATE"
  
  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.api_servers.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "principles-gym-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = true
  enable_http2              = true
  
  tags = {
    Environment = var.environment
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

### 2. Environment Configuration

Create `.env.production`:

```bash
# API Configuration
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/principles-gym/api.log

# Database Configuration
DATABASE_URL=postgresql+asyncpg://gymadmin:${DB_PASSWORD}@${DB_HOST}:5432/principles_gym
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_RECYCLE=3600
DATABASE_ECHO=false

# Redis Configuration
CACHE_REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:6379/0
CACHE_DEFAULT_TIMEOUT=3600

# Security
JWT_SECRET_KEY=${JWT_SECRET}
API_KEY=${API_KEY}
ENABLE_AUTH=true
CORS_ORIGINS=["https://yourdomain.com"]

# Performance Settings
MAX_SCENARIOS_PER_SESSION=1000
MAX_CONCURRENT_SESSIONS=500
ACTION_BUFFER_SIZE=10000
INFERENCE_BATCH_SIZE=100
INFERENCE_INTERVAL_SECONDS=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# AI Provider Keys (encrypted at rest)
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

### 3. Database Migration

```bash
# Initialize database schema
cd /opt/principles-gym
source venv/bin/activate

# Run Alembic migrations
alembic upgrade head

# Create indexes for performance
psql $DATABASE_URL << EOF
-- Composite indexes for efficient queries
CREATE INDEX CONCURRENTLY idx_actions_agent_timestamp 
ON actions(agent_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_principles_agent_active 
ON principles(agent_id, is_active) 
WHERE is_active = true;

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_actions_recent 
ON actions(timestamp) 
WHERE timestamp > NOW() - INTERVAL '7 days';

-- GIN index for JSONB searches
CREATE INDEX CONCURRENTLY idx_actions_metadata 
ON actions USING GIN (metadata);

-- Analyze tables for query optimizer
ANALYZE actions;
ANALYZE principles;
EOF
```

### 4. Deploy with Docker Swarm

```bash
# Initialize Swarm on manager node
docker swarm init --advertise-addr <MANAGER-IP>

# Join worker nodes
docker swarm join --token <WORKER-TOKEN> <MANAGER-IP>:2377

# Create overlay network
docker network create --driver overlay --attachable gym-network

# Deploy secrets
echo "$DB_PASSWORD" | docker secret create db_password -
echo "$JWT_SECRET" | docker secret create jwt_secret -
echo "$REDIS_PASSWORD" | docker secret create redis_password -

# Deploy stack
docker stack deploy -c docker-compose.prod.yml principles-gym
```

### 5. Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: principles-gym-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: principles-gym-api
  template:
    metadata:
      labels:
        app: principles-gym-api
    spec:
      containers:
      - name: api
        image: your-registry/principles-gym:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: principles-gym-api
  namespace: production
spec:
  selector:
    app: principles-gym-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: principles-gym-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: principles-gym-api
  minReplicas: 2
  maxReplicas: 10
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
```

## Scaling Considerations

### Horizontal Scaling Strategy

#### API Servers
- **Auto-scaling triggers**:
  - CPU > 70% for 5 minutes
  - Memory > 80% for 5 minutes
  - Request queue depth > 100
  - Response time p95 > 500ms
- **Scale-out**: Add 2 instances at a time
- **Scale-in**: Remove 1 instance at a time (cooldown: 10 min)
- **Maximum instances**: 20 per region

#### Database Scaling
- **Read Replicas**:
  - Add replicas when read QPS > 5000
  - Distribute reads using connection pooling
  - Implement read-after-write consistency
- **Partitioning Strategy**:
  - Partition actions table by month
  - Archive data older than 6 months
  - Use table inheritance for transparent access

### Queue Architecture for Training Jobs

Create dedicated queue workers:

```python
# queue_worker.py
import asyncio
from celery import Celery
from kombu import Queue

app = Celery('principles_gym')
app.config_from_object('celeryconfig')

# Define queues with priorities
app.conf.task_routes = {
    'training.high_priority': {'queue': 'high'},
    'training.normal': {'queue': 'normal'},
    'training.batch': {'queue': 'batch'}
}

app.conf.task_queues = (
    Queue('high', priority=10),
    Queue('normal', priority=5),
    Queue('batch', priority=1),
)

@app.task(bind=True, max_retries=3)
def process_training_session(self, session_id: str):
    """Process training session asynchronously"""
    try:
        # Training logic here
        pass
    except Exception as exc:
        # Exponential backoff retry
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

### CDN Configuration

```nginx
# CDN origin configuration
location /static/ {
    proxy_pass http://api-servers;
    proxy_cache static_cache;
    proxy_cache_valid 200 7d;
    proxy_cache_valid 404 1m;
    add_header X-Cache-Status $upstream_cache_status;
    
    # Cache based on Accept-Encoding
    proxy_cache_key "$scheme$request_method$host$request_uri$http_accept_encoding";
    
    # Enable gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json;
}

# API responses with short cache
location /api/reports/ {
    proxy_pass http://api-servers;
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_methods GET HEAD;
    proxy_cache_bypass $http_authorization;
}
```

### Rate Limiting Architecture

```nginx
# nginx rate limiting configuration
http {
    # Define rate limit zones
    limit_req_zone $binary_remote_addr zone=general:10m rate=60r/m;
    limit_req_zone $binary_remote_addr zone=training:10m rate=10r/m;
    limit_req_zone $binary_remote_addr zone=expensive:10m rate=5r/m;
    
    # Connection limits
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    
    server {
        # General API endpoints
        location /api/ {
            limit_req zone=general burst=20 nodelay;
            limit_conn addr 10;
            proxy_pass http://api-servers;
        }
        
        # Training endpoints (expensive operations)
        location /api/training/start {
            limit_req zone=training burst=5 nodelay;
            limit_conn addr 2;
            proxy_pass http://api-servers;
        }
        
        # Report generation (CPU intensive)
        location /api/reports/generate {
            limit_req zone=expensive burst=2 nodelay;
            limit_conn addr 1;
            proxy_pass http://api-servers;
        }
    }
}
```

## Monitoring Setup

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Load rules
rule_files:
  - "alerts/*.yml"

# Scrape configurations
scrape_configs:
  # API metrics
  - job_name: 'principles-gym-api'
    static_configs:
      - targets: ['api-1:9090', 'api-2:9090']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+):.*'
        replacement: '${1}'

  # PostgreSQL exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Node exporters
  - job_name: 'node'
    static_configs:
      - targets: ['node-1:9100', 'node-2:9100']
```

### Alert Rules

Create `alerts/api.yml`:

```yaml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(api_requests_total{status=~"5.."}[5m]) 
          / rate(api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

      # Slow response times
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95, 
            rate(api_response_time_seconds_bucket[5m])
          ) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow API response times"
          description: "95th percentile response time is {{ $value }}s"

      # High memory usage
      - alert: HighMemoryUsage
        expr: |
          process_resident_memory_bytes 
          / node_memory_MemTotal_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      # Database connection pool exhaustion
      - alert: DatabasePoolExhausted
        expr: |
          database_pool_size - database_pool_checkedout < 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool near exhaustion"
          description: "Only {{ $value }} connections available"
```

### Grafana Dashboards

Create comprehensive dashboards for:

1. **System Overview**
   - Request rate and error rate
   - Response time percentiles (p50, p95, p99)
   - Active training sessions
   - Principle discovery rate

2. **Performance Metrics**
   - CPU and memory usage by service
   - Database query performance
   - Cache hit rates
   - Queue depths and processing times

3. **Business Metrics**
   - Active agents by framework
   - Training sessions per hour
   - Principles discovered per day
   - Average behavioral entropy

### ELK Stack Configuration

```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON logs
  json {
    source => "message"
  }
  
  # Extract custom fields
  mutate {
    add_field => {
      "request_id" => "%{[request_id]}"
      "agent_id" => "%{[agent_id]}"
      "session_id" => "%{[session_id]}"
    }
  }
  
  # Parse response time
  if [response_time] {
    mutate {
      convert => { "response_time" => "float" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "principles-gym-%{+YYYY.MM.dd}"
  }
}
```

### Uptime Monitoring

Configure external monitoring using UptimeRobot or Pingdom:

```yaml
monitors:
  - name: "API Health Check"
    url: "https://api.principlesgym.com/health"
    interval: 60
    timeout: 30
    alerts:
      - email: "ops@company.com"
      - slack: "#alerts"
    
  - name: "WebSocket Endpoint"
    url: "wss://api.principlesgym.com/ws"
    type: "websocket"
    interval: 300
    
  - name: "Training Endpoint"
    url: "https://api.principlesgym.com/api/training/status/test"
    interval: 300
    expected_status: [200, 404]
```

### PagerDuty Integration

```python
# pagerduty_integration.py
import pypd
from typing import Dict, Any

class PagerDutyIntegration:
    def __init__(self, api_key: str, service_id: str):
        pypd.api_key = api_key
        self.service_id = service_id
    
    def create_incident(self, 
                       title: str, 
                       details: Dict[str, Any],
                       urgency: str = "high") -> str:
        """Create PagerDuty incident"""
        incident = pypd.Incident.create(
            service_id=self.service_id,
            title=title,
            body={
                "type": "incident_body",
                "details": details
            },
            urgency=urgency
        )
        return incident["id"]
    
    def auto_escalate(self, metric: str, value: float, threshold: float):
        """Auto-escalate based on metrics"""
        if metric == "error_rate" and value > 0.10:
            self.create_incident(
                f"Critical: Error rate at {value:.2%}",
                {"metric": metric, "value": value, "threshold": threshold},
                urgency="high"
            )
        elif metric == "response_time_p99" and value > 2.0:
            self.create_incident(
                f"High response times: p99 at {value:.2f}s",
                {"metric": metric, "value": value, "threshold": threshold},
                urgency="high"
            )
```

## Security Hardening

### 1. API Key Rotation Policy

```python
# api_key_rotation.py
import asyncio
from datetime import datetime, timedelta
from typing import List
import secrets

class APIKeyRotation:
    def __init__(self, db_manager, notification_service):
        self.db = db_manager
        self.notifier = notification_service
    
    async def rotate_keys(self, days_before_expiry: int = 7):
        """Rotate API keys before expiration"""
        expiry_threshold = datetime.utcnow() + timedelta(days=days_before_expiry)
        
        # Find keys expiring soon
        expiring_keys = await self.db.get_expiring_keys(expiry_threshold)
        
        for key_record in expiring_keys:
            # Generate new key
            new_key = secrets.token_urlsafe(32)
            
            # Create new key with extended expiration
            await self.db.create_api_key(
                user_id=key_record.user_id,
                key=new_key,
                expires_at=datetime.utcnow() + timedelta(days=90)
            )
            
            # Notify user
            await self.notifier.send_key_rotation_notice(
                user_id=key_record.user_id,
                old_key_hint=key_record.key[:8] + "...",
                new_key=new_key,
                expiry_date=key_record.expires_at
            )
    
    async def revoke_compromised_keys(self, key_patterns: List[str]):
        """Immediately revoke compromised keys"""
        for pattern in key_patterns:
            await self.db.revoke_keys_matching(pattern)
        
        # Force cache invalidation
        await self.cache.delete_pattern("auth:*")
```

### 2. Database Encryption

```sql
-- Enable encryption at rest
ALTER DATABASE principles_gym SET encryption = 'on';

-- Encrypt sensitive columns
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt API keys
ALTER TABLE api_keys 
ADD COLUMN key_encrypted bytea;

UPDATE api_keys 
SET key_encrypted = pgp_sym_encrypt(key, current_setting('app.encryption_key'));

ALTER TABLE api_keys 
DROP COLUMN key;

ALTER TABLE api_keys 
RENAME COLUMN key_encrypted TO key;

-- Create secure view for decryption
CREATE VIEW api_keys_decrypted AS
SELECT 
    id,
    user_id,
    pgp_sym_decrypt(key, current_setting('app.encryption_key')) as key,
    expires_at
FROM api_keys;
```

### 3. Network Isolation

```yaml
# docker-compose.security.yml
version: '3.9'

networks:
  frontend:
    driver: bridge
    internal: false
  backend:
    driver: bridge
    internal: true
  data:
    driver: bridge
    internal: true

services:
  nginx:
    networks:
      - frontend
      - backend
  
  api:
    networks:
      - backend
      - data
  
  postgres:
    networks:
      - data
  
  redis:
    networks:
      - data
```

### 4. Regular Security Updates

Create `security-updates.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Update system packages
apt-get update
apt-get upgrade -y

# Update Python dependencies
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Security audit
pip-audit --desc
safety check

# Update Docker images
docker pull postgres:16-alpine
docker pull redis:7-alpine
docker pull nginx:alpine

# Restart services with new images
docker-compose down
docker-compose up -d

# Run security tests
pytest tests/test_security.py -v
```

### 5. Penetration Testing Checklist

#### API Security
- [ ] Test for SQL injection in all endpoints
- [ ] Verify rate limiting effectiveness
- [ ] Check for XXE vulnerabilities in XML parsing
- [ ] Test JWT token validation and expiration
- [ ] Verify CORS configuration
- [ ] Test for path traversal vulnerabilities
- [ ] Check for command injection possibilities
- [ ] Verify input validation on all fields

#### Infrastructure Security
- [ ] Port scanning and service enumeration
- [ ] SSL/TLS configuration audit
- [ ] Network segmentation verification
- [ ] Firewall rule review
- [ ] SSH key rotation and audit
- [ ] Container security scanning
- [ ] Secrets management audit
- [ ] Backup encryption verification

## Operational Procedures

### 1. Deployment Runbook

#### Pre-deployment Checklist
```bash
#!/bin/bash
# pre-deploy.sh

echo "=== Pre-deployment Checklist ==="

# 1. Check system health
echo "Checking system health..."
curl -s https://api.principlesgym.com/health | jq .

# 2. Verify database connectivity
echo "Checking database..."
psql $DATABASE_URL -c "SELECT version();"

# 3. Check Redis connectivity
echo "Checking Redis..."
redis-cli -h $REDIS_HOST ping

# 4. Run test suite
echo "Running tests..."
pytest tests/ -v --tb=short

# 5. Check disk space
echo "Checking disk space..."
df -h | grep -E "(^Filesystem|/$|/var|/opt)"

# 6. Verify backups
echo "Checking latest backup..."
aws s3 ls s3://principles-gym-backups/ --recursive | tail -5

echo "=== Pre-deployment checks complete ==="
```

#### Zero-downtime Deployment Process
```bash
#!/bin/bash
# deploy.sh

set -euo pipefail

VERSION=$1
ENVIRONMENT=${2:-production}

echo "Deploying version $VERSION to $ENVIRONMENT"

# 1. Pull new image
docker pull your-registry/principles-gym:$VERSION

# 2. Run database migrations
docker run --rm \
  -e DATABASE_URL=$DATABASE_URL \
  your-registry/principles-gym:$VERSION \
  alembic upgrade head

# 3. Update service with rolling deployment
if [ "$ENVIRONMENT" == "production" ]; then
  # Rolling update for Kubernetes
  kubectl set image deployment/principles-gym-api \
    api=your-registry/principles-gym:$VERSION \
    --record

  # Wait for rollout
  kubectl rollout status deployment/principles-gym-api
else
  # Docker Swarm update
  docker service update \
    --image your-registry/principles-gym:$VERSION \
    --update-parallelism 1 \
    --update-delay 30s \
    principles-gym_api
fi

# 4. Verify deployment
sleep 30
./health-check.sh

echo "Deployment complete!"
```

### 2. Backup and Recovery

#### Automated Backup Strategy
```python
# backup_manager.py
import os
import asyncio
from datetime import datetime, timedelta
import boto3
import asyncpg

class BackupManager:
    def __init__(self, db_url: str, s3_bucket: str):
        self.db_url = db_url
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
    
    async def backup_database(self):
        """Create and upload database backup"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_principles_gym_{timestamp}.sql"
        
        # Create backup
        os.system(f"pg_dump {self.db_url} > {backup_file}")
        
        # Compress
        os.system(f"gzip {backup_file}")
        backup_file_gz = f"{backup_file}.gz"
        
        # Upload to S3
        with open(backup_file_gz, 'rb') as f:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"database/{backup_file_gz}",
                Body=f,
                ServerSideEncryption='AES256'
            )
        
        # Clean up local file
        os.remove(backup_file_gz)
        
        # Verify backup
        return await self.verify_backup(f"database/{backup_file_gz}")
    
    async def verify_backup(self, s3_key: str) -> bool:
        """Verify backup integrity"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.s3_bucket,
                Key=s3_key
            )
            return response['ContentLength'] > 1000  # Basic size check
        except Exception:
            return False
    
    async def cleanup_old_backups(self, retention_days: int = 30):
        """Remove backups older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=self.s3_bucket,
            Prefix='database/'
        )
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.s3_bucket,
                        Key=obj['Key']
                    )
```

#### Recovery Procedures
```bash
#!/bin/bash
# recover.sh

BACKUP_DATE=$1
S3_BUCKET="principles-gym-backups"

echo "=== Starting Recovery Process ==="

# 1. Download backup
echo "Downloading backup from $BACKUP_DATE..."
aws s3 cp s3://$S3_BUCKET/database/backup_principles_gym_$BACKUP_DATE.sql.gz .

# 2. Decompress
gunzip backup_principles_gym_$BACKUP_DATE.sql.gz

# 3. Stop API services
echo "Stopping API services..."
kubectl scale deployment principles-gym-api --replicas=0

# 4. Create recovery database
echo "Creating recovery database..."
psql $DATABASE_URL << EOF
CREATE DATABASE principles_gym_recovery;
EOF

# 5. Restore backup
echo "Restoring backup..."
psql principles_gym_recovery < backup_principles_gym_$BACKUP_DATE.sql

# 6. Verify data integrity
echo "Verifying data..."
psql principles_gym_recovery << EOF
SELECT COUNT(*) as agents FROM agent_profiles;
SELECT COUNT(*) as actions FROM actions;
SELECT COUNT(*) as principles FROM principles;
EOF

# 7. Switch databases
echo "Switching databases..."
psql $DATABASE_URL << EOF
ALTER DATABASE principles_gym RENAME TO principles_gym_old;
ALTER DATABASE principles_gym_recovery RENAME TO principles_gym;
EOF

# 8. Restart services
echo "Restarting services..."
kubectl scale deployment principles-gym-api --replicas=3

echo "=== Recovery Complete ==="
```

### 3. Incident Response

#### Incident Response Playbook
```yaml
# incident_response.yaml
incidents:
  high_error_rate:
    severity: P1
    escalation_time: 5m
    steps:
      - check: "Verify error rate in Grafana dashboard"
      - action: "Check recent deployments"
      - action: "Review error logs in Kibana"
      - action: "Check database connectivity"
      - action: "Verify external API availability"
      - decision:
          if_deployment_issue: "Rollback to previous version"
          if_database_issue: "Failover to replica"
          if_external_api: "Enable circuit breaker"
    
  database_outage:
    severity: P1
    escalation_time: 2m
    steps:
      - check: "Verify database status"
      - action: "Attempt automatic failover"
      - action: "Check connection pool status"
      - action: "Review database logs"
      - decision:
          if_primary_down: "Promote read replica"
          if_connection_exhausted: "Restart connection pools"
          if_corruption: "Restore from backup"
  
  memory_leak:
    severity: P2
    escalation_time: 15m
    steps:
      - check: "Identify affected instances"
      - action: "Capture heap dump"
      - action: "Analyze memory patterns"
      - decision:
          if_gradual_increase: "Schedule rolling restart"
          if_rapid_increase: "Immediate instance replacement"
```

#### Automated Response Scripts
```python
# incident_automation.py
import asyncio
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class IncidentResponder:
    def __init__(self, monitoring_client, deployment_client, notification_client):
        self.monitoring = monitoring_client
        self.deployment = deployment_client
        self.notifier = notification_client
    
    async def handle_high_error_rate(self, error_rate: float):
        """Automated response to high error rates"""
        logger.info("high_error_rate_detected", rate=error_rate)
        
        # 1. Check if recent deployment
        recent_deploy = await self.deployment.get_recent_deployment()
        if recent_deploy and recent_deploy.age_minutes < 30:
            logger.info("recent_deployment_detected", 
                       version=recent_deploy.version)
            
            # Auto-rollback if error rate > 10%
            if error_rate > 0.10:
                await self.deployment.rollback()
                await self.notifier.send_alert(
                    "Auto-rollback initiated due to high error rate",
                    severity="critical"
                )
                return
        
        # 2. Check external dependencies
        deps_healthy = await self.check_dependencies()
        if not deps_healthy:
            # Enable circuit breakers
            await self.enable_circuit_breakers()
            await self.notifier.send_alert(
                "Circuit breakers enabled due to dependency failures",
                severity="warning"
            )
    
    async def handle_database_issue(self, issue_type: str):
        """Automated database issue response"""
        if issue_type == "connection_pool_exhausted":
            # Restart affected instances
            await self.deployment.restart_unhealthy_instances()
        
        elif issue_type == "replica_lag":
            # Route traffic away from lagged replica
            await self.deployment.update_database_routing(
                exclude_replicas=["replica-2"]
            )
        
        elif issue_type == "primary_unresponsive":
            # Initiate failover
            await self.database_failover()
```

### 4. Performance Tuning

#### Database Optimization
```sql
-- Regular maintenance tasks
-- Run weekly during low-traffic periods

-- Update statistics
ANALYZE;

-- Rebuild indexes
REINDEX CONCURRENTLY INDEX idx_actions_agent_timestamp;
REINDEX CONCURRENTLY INDEX idx_principles_agent_active;

-- Vacuum tables
VACUUM (ANALYZE, VERBOSE) actions;
VACUUM (ANALYZE, VERBOSE) principles;

-- Monitor slow queries
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Top 10 slowest queries
SELECT 
    round(total_exec_time::numeric, 2) AS total_time_ms,
    calls,
    round(mean_exec_time::numeric, 2) AS mean_time_ms,
    round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) AS percentage,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

#### Application Performance
```python
# performance_tuning.py
from typing import Dict, Any
import asyncio
import aioredis
from functools import lru_cache

class PerformanceOptimizer:
    def __init__(self, cache_client: aioredis.Redis):
        self.cache = cache_client
    
    async def optimize_caching(self):
        """Analyze and optimize cache usage"""
        # Get cache statistics
        info = await self.cache.info()
        
        hit_rate = info['keyspace_hits'] / (
            info['keyspace_hits'] + info['keyspace_misses']
        )
        
        if hit_rate < 0.8:
            # Adjust cache strategy
            await self.increase_cache_ttl()
            await self.implement_cache_warming()
    
    @lru_cache(maxsize=1000)
    def optimize_pattern_matching(self, sequence: tuple) -> float:
        """Cache expensive DTW calculations"""
        # DTW calculation cached in memory
        return calculate_dtw(sequence)
    
    async def optimize_database_queries(self):
        """Implement query optimization strategies"""
        # Use prepared statements
        self.prepared_statements = {
            'get_recent_actions': """
                SELECT * FROM actions 
                WHERE agent_id = $1 
                AND timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT $2
            """,
            'get_active_principles': """
                SELECT * FROM principles
                WHERE agent_id = $1 AND is_active = true
                ORDER BY strength DESC
            """
        }
```

### 5. Capacity Planning

#### Growth Projections
```python
# capacity_planning.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple

class CapacityPlanner:
    def __init__(self, metrics_client):
        self.metrics = metrics_client
    
    async def project_growth(self, days_ahead: int = 90) -> Dict[str, float]:
        """Project resource needs based on growth trends"""
        # Get historical data
        history = await self.metrics.get_historical_data(
            metrics=['requests_per_second', 'active_agents', 'storage_used'],
            days_back=90
        )
        
        # Calculate growth rates
        growth_rates = {}
        for metric, data in history.items():
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Linear regression for growth rate
            x = (df['timestamp'] - df['timestamp'].min()).dt.days
            y = df['value']
            
            slope = np.polyfit(x, y, 1)[0]
            growth_rates[metric] = slope
        
        # Project future needs
        projections = {}
        for metric, rate in growth_rates.items():
            current = history[metric][-1]['value']
            projected = current + (rate * days_ahead)
            projections[metric] = projected
        
        return self.calculate_resource_needs(projections)
    
    def calculate_resource_needs(self, projections: Dict[str, float]) -> Dict[str, Any]:
        """Convert projections to resource requirements"""
        return {
            'api_instances': max(2, int(projections['requests_per_second'] / 500)),
            'database_cpu': max(8, int(projections['active_agents'] / 1000) * 2),
            'database_storage_gb': int(projections['storage_used'] * 1.5),
            'redis_memory_gb': max(8, int(projections['active_agents'] / 5000) * 8)
        }
```

### 6. Maintenance Windows

#### Scheduled Maintenance
```yaml
# maintenance_schedule.yaml
maintenance_windows:
  weekly:
    - name: "Database Statistics Update"
      schedule: "0 3 * * 0"  # Sunday 3 AM
      duration: "30m"
      tasks:
        - "ANALYZE all tables"
        - "Update pg_stat_statements"
        - "Clean old partitions"
    
    - name: "Cache Optimization"
      schedule: "0 4 * * 0"  # Sunday 4 AM
      duration: "15m"
      tasks:
        - "Analyze cache hit rates"
        - "Adjust TTL values"
        - "Clear stale entries"
  
  monthly:
    - name: "Security Updates"
      schedule: "0 2 15 * *"  # 15th of month, 2 AM
      duration: "2h"
      tasks:
        - "System package updates"
        - "Docker image updates"
        - "SSL certificate renewal check"
        - "Security scan"
    
    - name: "Performance Review"
      schedule: "0 1 1 * *"  # 1st of month, 1 AM
      duration: "1h"
      tasks:
        - "Index rebuild"
        - "Query plan analysis"
        - "Resource utilization review"
```

#### Maintenance Mode Implementation
```python
# maintenance_mode.py
import asyncio
from datetime import datetime
from typing import Optional

class MaintenanceMode:
    def __init__(self, redis_client, notification_service):
        self.redis = redis_client
        self.notifier = notification_service
    
    async def enable(self, 
                    reason: str, 
                    duration_minutes: int,
                    readonly: bool = False):
        """Enable maintenance mode"""
        maintenance_info = {
            'enabled': True,
            'reason': reason,
            'started_at': datetime.utcnow().isoformat(),
            'expected_end': (
                datetime.utcnow() + 
                timedelta(minutes=duration_minutes)
            ).isoformat(),
            'readonly': readonly
        }
        
        # Set in Redis
        await self.redis.setex(
            'maintenance_mode',
            duration_minutes * 60,
            json.dumps(maintenance_info)
        )
        
        # Notify users
        await self.notifier.broadcast_maintenance(maintenance_info)
    
    async def disable(self):
        """Disable maintenance mode"""
        await self.redis.delete('maintenance_mode')
        await self.notifier.broadcast_maintenance_end()
    
    async def check_status(self) -> Optional[Dict[str, Any]]:
        """Check if maintenance mode is active"""
        data = await self.redis.get('maintenance_mode')
        return json.loads(data) if data else None
```

### 7. Documentation and Knowledge Base

#### Runbook Template
```markdown
# Runbook: [Service Name]

## Overview
Brief description of the service and its purpose.

## Architecture
- Components involved
- Dependencies
- Data flow

## Common Issues and Solutions

### Issue: High Memory Usage
**Symptoms:**
- Memory usage > 80%
- Slow response times
- OOM errors in logs

**Resolution:**
1. Check for memory leaks: `jmap -histo PID`
2. Review recent code changes
3. Increase memory limit if legitimate growth
4. Restart service if immediate relief needed

### Issue: Database Connection Errors
**Symptoms:**
- "connection pool exhausted" errors
- Timeouts on database operations

**Resolution:**
1. Check connection pool metrics
2. Review slow query log
3. Increase pool size if needed
4. Check for connection leaks

## Monitoring
- Dashboard: https://grafana.company.com/d/principles-gym
- Alerts: https://prometheus.company.com/alerts
- Logs: https://kibana.company.com

## Contacts
- On-call: #ops-oncall
- Engineering: #principles-gym-eng
- Escalation: See PagerDuty
```

This completes the comprehensive production deployment guide for AI Principles Gym, covering all aspects from infrastructure setup through operational procedures.
