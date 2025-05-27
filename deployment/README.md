# AI Principles Gym - Docker Deployment Guide

## Overview

This directory contains the Docker configuration for deploying the AI Principles Gym in production. The setup includes:

- **API Service**: The main FastAPI application
- **PostgreSQL**: Primary database for persistent storage
- **Redis**: Cache and session storage
- **Nginx**: Reverse proxy with rate limiting

## Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB free disk space

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-principles-gym/deployment
   ```

2. **Configure environment**:
   ```bash
   cp .env.production .env
   # Edit .env and update:
   # - JWT_SECRET_KEY (generate a strong secret)
   # - API_KEY (for service authentication)
   # - Database passwords
   # - CORS_ORIGINS
   ```

3. **Build and start services**:
   ```bash
   docker-compose up -d --build
   ```

4. **Check service health**:
   ```bash
   docker-compose ps
   curl http://localhost/health
   ```

## Service Details

### API Service
- **Port**: 8000 (internal), exposed via Nginx on port 80
- **Health Check**: `http://localhost/health`
- **Metrics**: `http://localhost/metrics` (restricted to internal network)
- **Resource Limits**: 2 CPU cores, 2GB RAM

### PostgreSQL Database
- **Port**: 5432
- **Database**: `principles_gym`
- **User**: `gymuser` (configure password in .env)
- **Data Volume**: `postgres-data`
- **Resource Limits**: 1 CPU core, 1GB RAM

### Redis Cache
- **Port**: 6379
- **Password**: Configure in docker-compose.yml
- **Persistence**: AOF enabled with per-second fsync
- **Max Memory**: 512MB with LRU eviction
- **Resource Limits**: 0.5 CPU cores, 512MB RAM

### Nginx Reverse Proxy
- **HTTP Port**: 80
- **HTTPS Port**: 443 (requires SSL certificate configuration)
- **Rate Limiting**:
  - General API: 60 requests/minute
  - API Key Generation: 5 requests/minute
  - Training Endpoints: 10 requests/minute
  - Report Endpoints: 30 requests/minute
- **Resource Limits**: 0.5 CPU cores, 256MB RAM

## Rate Limiting

The Nginx configuration implements multiple rate limiting zones:

- **General endpoints**: 60 req/min with burst of 10
- **API key generation**: 5 req/min with burst of 2
- **Training operations**: 10 req/min with burst of 5
- **Report retrieval**: 30 req/min with burst of 10

## SSL/TLS Configuration

For production deployments with HTTPS:

1. **Obtain SSL certificates** (e.g., from Let's Encrypt)
2. **Place certificates** in `nginx/ssl/`:
   ```
   nginx/ssl/cert.pem
   nginx/ssl/key.pem
   ```
3. **Update nginx configuration**:
   - Uncomment HTTPS server block in `nginx/conf.d/api.conf`
   - Update `server_name` with your domain
   - Enable HTTP to HTTPS redirect

## Monitoring

### Health Checks
All services include health checks:
- API: `curl http://localhost/health`
- PostgreSQL: `docker exec principles-gym-postgres pg_isready`
- Redis: `docker exec principles-gym-redis redis-cli ping`
- Nginx: Built-in configuration test

### Logs
View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f nginx
```

Log locations:
- API logs: `./logs/api.log`
- Nginx logs: Docker volume `nginx-logs`

### Metrics
The API exposes Prometheus-compatible metrics at `/metrics` (restricted to internal network).

## Scaling

### Horizontal Scaling
To run multiple API instances:

1. **Update docker-compose.yml**:
   ```yaml
   api:
     scale: 3  # Run 3 instances
   ```

2. **Nginx will automatically load balance** across instances

### Vertical Scaling
Adjust resource limits in docker-compose.yml `deploy` section.

## Backup and Recovery

### Database Backup
```bash
# Backup
docker exec principles-gym-postgres pg_dump -U gymuser principles_gym > backup.sql

# Restore
docker exec -i principles-gym-postgres psql -U gymuser principles_gym < backup.sql
```

### Redis Backup
Redis automatically persists data via AOF. Backup files are in the `redis-data` volume.

## Maintenance

### Update Services
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Database Migrations
```bash
# Run Alembic migrations
docker exec principles-gym-api alembic upgrade head
```

### Clean Up
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove old images
docker image prune -a
```

## Security Considerations

1. **Change default passwords** in production
2. **Use strong JWT secret key** (minimum 32 characters)
3. **Configure firewall** to restrict database ports
4. **Enable HTTPS** for production deployments
5. **Regularly update** base images and dependencies
6. **Monitor logs** for suspicious activity
7. **Implement backup strategy** for data persistence

## Troubleshooting

### Service won't start
- Check logs: `docker-compose logs <service>`
- Verify port availability: `netstat -tlnp`
- Check resource limits: `docker stats`

### Database connection errors
- Verify PostgreSQL is healthy: `docker-compose ps postgres`
- Check connection string in environment
- Ensure database migrations have run

### High memory usage
- Check buffer sizes in configuration
- Monitor with: `docker stats`
- Adjust resource limits if needed

### Rate limiting issues
- Check Nginx logs for 429 responses
- Adjust rate limits in `nginx/nginx.conf`
- Consider implementing API key authentication

## Performance Tuning

### PostgreSQL
- Adjust `shared_buffers` and `work_mem`
- Configure connection pooling
- Regular VACUUM and ANALYZE

### Redis
- Monitor memory usage
- Adjust eviction policy if needed
- Consider Redis Cluster for high load

### API
- Tune worker processes
- Adjust connection pool sizes
- Enable response caching where appropriate

## Support

For issues or questions:
1. Check application logs
2. Review health check endpoints
3. Consult the main project documentation
4. Submit issues to the project repository
