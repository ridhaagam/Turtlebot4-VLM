# Server Deployment Guide

Complete guide for deploying the Smart Waste Bin Monitoring System server using Docker.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Docker Installation](#docker-installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Verify Services](#verify-services)
6. [Model Download](#model-download)
7. [API Endpoints](#api-endpoints)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 20.04+ / Windows with WSL2 / macOS |
| CPU | 4+ cores recommended |
| RAM | 8GB minimum, 16GB recommended |
| GPU | NVIDIA GPU with CUDA support (recommended) |
| Storage | 10GB+ for models and data |
| Docker | Docker 20.10+ with Compose v2 |
| NVIDIA Driver | 525+ for GPU support |

### Without GPU

The server can run on CPU-only, but inference will be slower:
- Florence-2: ~2-5 seconds per image (vs ~0.1s on GPU)
- SmolVLM2: ~5-15 seconds per inference (vs ~0.5s on GPU)

---

## Docker Installation

### Ubuntu

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

# Add Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

### NVIDIA Container Toolkit (for GPU support)

```bash
# Add NVIDIA GPG key and repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify Docker Installation

```bash
docker --version
docker compose version
# With GPU: should show your GPU
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/embedded-system-final.git
cd embedded-system-final
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with secure passwords
nano .env
```

**Required changes in .env:**
```bash
# Database passwords (CHANGE THESE!)
DB_ROOT_PASSWORD=your-secure-root-password-here
DB_PASSWORD=your-secure-admin-password-here

# Flask secrets (CHANGE THESE!)
SECRET_KEY=your-random-secret-key-min-32-chars
JWT_SECRET=your-jwt-secret-key-min-32-chars
HMAC_SECRET=your-hmac-secret-key-min-32-chars

# Enable vision models
LOAD_VISION_MODEL=true
```

### 3. Start Services

```bash
# Navigate to docker directory
cd docker

# Build and start all services
docker compose up -d --build

# Watch logs
docker compose logs -f
```

### 4. Verify Running

```bash
# Check service status
docker compose ps

# Check health endpoint
curl http://localhost:5000/api/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "database": "connected",
  "vision_model": "loaded"
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_ROOT_PASSWORD` | - | MySQL root password (required) |
| `DB_PASSWORD` | - | MySQL admin user password (required) |
| `SECRET_KEY` | - | Flask session secret (required) |
| `JWT_SECRET` | - | JWT signing secret (required) |
| `HMAC_SECRET` | - | HMAC signing secret (required) |
| `FLASK_ENV` | production | Flask environment |
| `LOAD_VISION_MODEL` | true | Load Florence-2/SmolVLM models |

### Service Ports

| Service | Internal Port | External Port | Description |
|---------|---------------|---------------|-------------|
| Frontend | 80 | 3004 | React dashboard |
| Backend | 5000 | 5000 | Flask API server |
| MySQL | 3306 | 3303 | Database |

### Persistent Volumes

| Volume | Path | Description |
|--------|------|-------------|
| `mysql_data` | /var/lib/mysql | Database storage |
| `upload_data` | /app/uploads | Detection images |
| `huggingface_cache` | ~/.cache/huggingface | Model cache |

---

## Verify Services

### Check All Services

```bash
cd docker
docker compose ps
```

**Expected output:**
```
NAME                 STATUS          PORTS
detection-backend    Up (healthy)    0.0.0.0:5000->5000/tcp
detection-frontend   Up              0.0.0.0:3004->80/tcp
detection-mysql      Up (healthy)    0.0.0.0:3303->3306/tcp
```

### Test Backend API

```bash
# Health check
curl http://localhost:5000/api/health

# Login test (returns JWT token)
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

### Test Frontend

Open browser: http://localhost:3004

Default credentials:
- Username: `admin`
- Password: `admin`

### Test Database

```bash
docker compose exec mysql mysql -u admin -p -e "SHOW DATABASES;"
# Enter password when prompted
```

---

## Model Download

The vision models are automatically downloaded on first startup. This may take 5-10 minutes.

### Models Used

| Model | Size | Purpose |
|-------|------|---------|
| Florence-2-base | ~1GB | Object detection, bounding boxes |
| SmolVLM2 | ~2GB | Navigation commands, bin classification |

### Pre-download Models (Optional)

To avoid first-startup delay:

```bash
docker compose exec backend python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

print("Downloading Florence-2...")
AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16)

print("Models downloaded successfully!")
EOF
```

### Verify Models Loaded

```bash
curl http://localhost:5000/api/health
# Check: "vision_model": "loaded"
```

---

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Login, returns JWT + HMAC secret |
| `/api/auth/verify` | GET | Verify JWT token |
| `/api/auth/logout` | POST | Logout (client discards token) |

### Vision Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vision/detect` | POST | Florence-2 object detection |
| `/api/vision/navigate` | POST | Navigation direction inference |
| `/api/vision/classify` | POST | Bin fullness classification |
| `/api/camera/frame-infer` | POST | Upload frame with inference |

### Detections

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detections` | GET | List detections (paginated) |
| `/api/detections` | POST | Create detection record |
| `/api/detections/latest` | GET | Get most recent detection |
| `/api/detections/<id>` | GET | Get specific detection |
| `/api/detections/stats` | GET | Get statistics |

### Camera Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/camera/frame` | POST | Upload robot camera frame |
| `/api/camera/classification/frame` | POST | Upload classification camera |

---

## Production Deployment

### 1. Use Strong Secrets

Generate secure random keys:

```bash
# Generate 32-char random strings
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET
openssl rand -hex 32  # For HMAC_SECRET
openssl rand -hex 16  # For DB passwords
```

### 2. Enable HTTPS (Reverse Proxy)

Example nginx configuration for SSL termination:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3004;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /socket.io {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. Firewall Configuration

```bash
# Allow required ports
sudo ufw allow 3004/tcp  # Frontend
sudo ufw allow 5000/tcp  # Backend API
# DON'T expose MySQL (3303) publicly!
```

### 4. Resource Limits

Edit `docker-compose.yml` to add limits:

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Troubleshooting

### Backend Not Starting

```bash
# Check logs
docker compose logs backend

# Common issues:
# 1. MySQL not ready - wait for healthcheck
# 2. GPU not available - check nvidia-docker
# 3. Out of memory - increase RAM or disable vision model
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi

# If failed, reinstall nvidia-container-toolkit
```

### Database Connection Failed

```bash
# Check MySQL is running
docker compose ps mysql

# Check logs
docker compose logs mysql

# Reset database (WARNING: deletes all data)
docker compose down -v
docker compose up -d
```

### Model Loading Failed

```bash
# Check available memory
free -h

# Check GPU memory
nvidia-smi

# Disable vision model temporarily
# In .env: LOAD_VISION_MODEL=false
```

### Frontend Not Accessible

```bash
# Check frontend container
docker compose logs frontend

# Verify nginx is serving
docker compose exec frontend curl localhost:80
```

### Edge Device Can't Connect

1. Check server IP is correct in edge device config
2. Verify firewall allows port 5000
3. Test connection:
   ```bash
   # From edge device
   curl http://SERVER_IP:5000/api/health
   ```

---

## Useful Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f backend

# Restart single service
docker compose restart backend

# Rebuild after code changes
docker compose up -d --build

# Clean everything (including volumes)
docker compose down -v

# Enter backend container
docker compose exec backend bash

# Database shell
docker compose exec mysql mysql -u admin -p detections
```

---

## Version Information

Tested with:
- Docker 24.x
- Docker Compose v2.x
- NVIDIA Driver 525+
- CUDA 12.x
- Ubuntu 22.04 LTS

---

## Quick Reference

```bash
# Full setup
cd embedded-system-final
cp .env.example .env
nano .env  # Edit secrets
cd docker
docker compose up -d --build

# Check status
docker compose ps
curl http://localhost:5000/api/health

# View dashboard
open http://localhost:3004

# Login: admin / admin
```
