# Docker Deployment Guide for AI Risk Manager

## Overview

This document provides instructions for building, running, and deploying the AI Risk Manager application using Docker.

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 1.29+ (optional, for simplified orchestration)
- At least 2GB of available disk space
- 1GB+ available RAM for the container

## Quick Start

### Using docker-compose (Recommended)

```bash
# Build and start the application
docker-compose up --build

# Application will be available at http://localhost:8000
# Dashboard at http://localhost:8000/static/index.html
```

### Using Docker CLI

```bash
# Build the image
docker build -t ai-risk-manager:latest .

# Run the container
docker run -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -e FLASK_ENV=production \
  --name ai-risk-manager \
  ai-risk-manager:latest

# Stop the container
docker stop ai-risk-manager

# Remove the container
docker rm ai-risk-manager
```

## Image Details

### Base Image
- **Image**: python:3.10-slim
- **Size**: ~150MB (optimized with multi-stage build)
- **OS**: Debian Trixie

### Layers
1. **Builder Stage**: Compiles Python dependencies from requirements.txt
2. **Runtime Stage**: Copies only compiled dependencies, reducing final image size

### Environment Variables
- `PYTHONUNBUFFERED=1`: Stream logs to stdout in real-time
- `PYTHONDONTWRITEBYTECODE=1`: Don't create .pyc files
- `FLASK_ENV=production`: Run in production mode
- `FLASK_APP=backend.app`: Flask application entry point

## API Endpoints

Once running, the following endpoints are available:

- `GET /health` - Health check endpoint
- `POST /api/risk/analyze` - Analyze single merchant-day
- `GET /api/risk/merchants` - List merchants
- `GET /api/risk/timeseries/<merchant_id>` - Historical data
- `GET /api/risk/portfolio` - Portfolio analysis
- `GET /api/risk/audit` - Audit log
- `GET /api/risk/evaluation` - Model evaluation metrics
- `GET /static/index.html` - Dashboard UI

## Health Check

The container includes a built-in health check:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

## Logs

View container logs:

```bash
# Using docker-compose
docker-compose logs -f

# Using Docker CLI
docker logs -f ai-risk-manager
```

## Performance Tuning

### Resource Limits

In `docker-compose.yml`, you can configure resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

### Persistence

To persist data across container restarts, mount a volume:

```bash
docker run -p 8000:8000 \
  -v ai_risk_data:/app/data \
  ai-risk-manager:latest
```

## Debugging

### Interactive Shell

```bash
# Start container with interactive shell
docker run -it ai-risk-manager:latest /bin/bash

# Or exec into running container
docker exec -it ai-risk-manager /bin/bash
```

### Run Tests in Container

```bash
docker run ai-risk-manager:latest python -m pytest tests/ -v
```

### Inspect Image

```bash
docker inspect ai-risk-manager:latest
```

## Deployment Scenarios

### Local Development

```bash
docker-compose up --build
```

### Production on Cloud (AWS, GCP, Azure)

1. Push image to container registry:
```bash
docker tag ai-risk-manager:latest your-registry/ai-risk-manager:v1.0
docker push your-registry/ai-risk-manager:v1.0
```

2. Deploy using Kubernetes or container orchestration service

### Production with Nginx Reverse Proxy

See `docker-compose.nginx.yml` for production configuration with Nginx.

## Troubleshooting

### Container fails to start

Check logs:
```bash
docker logs ai-risk-manager
```

### Port 8000 already in use

Use a different port:
```bash
docker run -p 9000:8000 ai-risk-manager:latest
```

### Out of memory errors

Increase available memory to container or system:
```bash
docker run -m 2g ai-risk-manager:latest
```

### Build fails with dependency errors

Ensure Docker has internet access and requirements.txt is valid:
```bash
docker build --progress=plain -t ai-risk-manager:latest .
```

## Security Considerations

1. **Run as non-root user** (current setup uses default Python user)
2. **Use secrets management** for sensitive data (not included in current build)
3. **Network isolation**: Use Docker networks to isolate containers
4. **Image scanning**: Regularly scan image for vulnerabilities
5. **Keep base image updated**: Pull latest python:3.10-slim regularly

## Maintenance

### Cleaning up

```bash
# Remove dangling images
docker image prune

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune
```

### Updating Application

1. Make code changes
2. Rebuild image: `docker build -t ai-risk-manager:latest .`
3. Restart container: `docker-compose up --build`

## Testing in Docker

```bash
# Run tests
docker-compose exec ai-risk-manager python -m pytest tests/ -v

# Or using docker run
docker run ai-risk-manager:latest python -m pytest tests/ -v
```

## Next Steps

- Review [README.md](README.md) for application overview
- Check [requirements.txt](requirements.txt) for dependencies
- Explore API documentation via `/api/risk/analyze` endpoint
- Review test suite in `tests/` directory
