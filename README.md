# Smart Waste Bin Monitoring System

A real-time waste bin monitoring system using **Florence-2 Vision Language Model** for intelligent bin detection and fullness classification. Designed for **TurtleBot4 Lite** autonomous robot with **Raspberry Pi 4** or **NVIDIA Jetson Orin** edge devices.

## System Architecture

![System Architecture](embeddedsystem-bin.png)

---

## Setup Guides

| Platform | Guide | Description |
|----------|-------|-------------|
| **Server** | [SERVER_SETUP.md](SERVER_SETUP.md) | Docker deployment for dashboard and vision inference |
| **Raspberry Pi 4** | [RPI_SETUP.md](RPI_SETUP.md) | ROS2 Jazzy + TurtleBot4 Lite robot setup |
| **Jetson Orin** | [JETSON_SETUP.md](JETSON_SETUP.md) | Edge device with local Florence-2 inference |
| **Robot (edge-bot)** | ROS2 package development guide |

### Quick Start Order

1. **Deploy Server** - Set up dashboard and vision API (see [SERVER_SETUP.md](SERVER_SETUP.md))
2. **Setup Robot** - Configure TurtleBot4 with ROS2 (see [RPI_SETUP.md](RPI_SETUP.md))
3. **Run Mission** - Start autonomous bin search

---

## Model Download

The vision models are automatically downloaded on first run. To pre-download:

### On Server (Docker)
```bash
docker compose exec backend python3 -c "
from transformers import AutoModelForCausalLM, AutoProcessor
AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
print('Florence-2 downloaded!')
"
```

### On Jetson (Local Inference)
```bash
./scripts/setup_jetson.sh  # Includes model download
```

### Model Sizes
| Model | Size | Purpose |
|-------|------|---------|
| Florence-2-base | ~1GB | Object detection |
| SmolVLM2 | ~2GB | Navigation & classification |

---

## Features

- **Real-time Object Detection**: Uses Florence-2 VLM for accurate object detection
- **Bin Fullness Classification**: Classifies bins as Empty (0-25%), Partial (25-75%), Mostly Full (75-90%), or Full (90-100%)
- **Content Detection Inside Bins**: Detects and displays objects inside bins with separate bounding boxes
- **Visual Question Answering**: Uses VQA to analyze bin contents for accurate fullness estimation
- **VLM-Generated Summaries**: Florence-2 generates natural language summaries of bin status
- **Beautiful Dashboard**: Modern React dashboard with atmospheric aesthetics, animations, and real-time updates
- **Secure Authentication**: JWT-based login system with HMAC-SHA256 request signing
- **Dual Image Display**: Shows both original image and annotated detection view
- **WebSocket Support**: Real-time updates via Socket.IO for instant feedback
- **Cross-Platform**: Supports Jetson Orin (CUDA) and Raspberry Pi 4 (CPU)
- **Paginated History**: Browse through detection history with full pagination support

## Detailed Architecture

```
                            EDGE DEVICE (Jetson Orin / RPi 4)
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────────────────┐│
│  │ USB Cam  │───▶│ Frame Capture│───▶│ Florence-2 Vision Language Model   ││
│  │          │    │  (OpenCV)    │    │  - Object Detection (<OD>)         ││
│  └──────────┘    └──────────────┘    │  - VQA for Fullness Classification ││
│                                      └──────────────┬──────────────────────┘│
│                                                     │                        │
│                         ┌───────────────────────────▼───────────────────────┐│
│                         │  Detection Results + Bin Fullness Classification ││
│                         │  - Bounding boxes, labels, confidence             ││
│                         │  - Per-bin fullness: 0-25%, 25-75%, 75-90%, 90%+  ││
│                         └───────────────────────────┬───────────────────────┘│
└─────────────────────────────────────────────────────│────────────────────────┘
                                                      │ REST API + Images
                                                      ▼
                              DASHBOARD SERVER (Laptop/Server)
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────────────────┐   ┌──────────────┐   ┌─────────────────────────────┐│
│  │ Flask Backend       │   │              │   │ React Dashboard             ││
│  │ + Socket.IO         │◀─▶│   MySQL DB   │◀─▶│ - Live detection feed       ││
│  │ + Image Storage     │   │              │   │ - Original + BBox view      ││
│  └─────────────────────┘   └──────────────┘   │ - Fullness gauge & summary  ││
│                                               │ - Detection history         ││
│  Ports:                                       │ - Statistics & analytics    ││
│  - MySQL: 3303                                └─────────────────────────────┘│
│  - Backend: 5000                                                             │
│  - Frontend: 3004                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Dashboard (on your laptop)

```bash
# Clone the repository
git clone <repo-url>
cd embedded-system-final

# Start everything with one command
./scripts/build_dashboard.sh

# Dashboard available at:
# - Frontend: http://localhost:3004
# - API: http://localhost:5000
```

### 2. Start Edge Detection (on Jetson/RPi)

```bash
# Copy entire project to your device, then:

# First time setup (installs dependencies + downloads model ~1GB)
./scripts/setup_edge.sh

# Edit config to set your laptop's IP
nano edge/config/config.yaml
# Change: url: "http://YOUR_LAPTOP_IP:5000"

# Run detection
./scripts/build_edge.sh run
```

## Bin Fullness Classification

The system uses a multi-step approach to classify bin fullness:

1. **Object Detection**: Florence-2 detects bins, trash cans, waste containers
2. **Visual Question Answering**: For each detected bin, VQA analyzes:
   - How full the bin appears
   - Content visibility
   - Visual fill level estimation
3. **Fallback Analysis**: If VQA fails, uses visual heuristics (color variance)

### Fullness Levels

| Level | Range | Color | Action |
|-------|-------|-------|--------|
| Empty | 0-25% | Green | Good capacity available |
| Partial | 25-75% | Yellow-Green | Moderate capacity |
| Mostly Full | 75-90% | Orange | Schedule pickup soon |
| Full | 90-100% | Red | Immediate attention needed |

## Dashboard Features

### Live Detection Feed
- **Dual panel view**: Original image alongside annotated detection view
- **Real-time bounding boxes**: Color-coded by object type and fullness level
- **Corner accent styling**: Modern, sophisticated detection overlays

### Bin Fullness Card
- **Large percentage display**: Instant visibility of overall fullness
- **Animated gauge bar**: Smooth transitions with shimmer effect
- **Per-bin indicators**: Individual status for multiple bins

### Summary Card
- **Situation summary**: Human-readable description of current state
- **Status badges**: Quick indicators for detection, bins, and performance
- **Meta information**: Timestamp, inference time, device info

### Statistics Panel
- **Total frames and objects**: Overall detection counts
- **Bins found**: Total waste containers detected
- **Average fullness**: Running average across all detections
- **Fullness distribution**: Visual breakdown by category
- **Top detected labels**: Most common objects

### Detection History
- **Chronological list**: Browse past detections
- **Fullness bars**: Quick visual indicator per detection
- **Click to view**: Select any detection for detailed view

## Project Structure

```
embedded-system-final/
├── edge/                          # Edge device code (Jetson/RPi)
│   ├── src/
│   │   ├── platform.py            # Platform detection
│   │   ├── camera.py              # USB camera capture
│   │   ├── detector.py            # Florence-2 + VQA inference
│   │   ├── api_client.py          # REST/WebSocket client
│   │   └── main.py                # Main entry point
│   ├── config/
│   │   ├── config.yaml            # Base configuration
│   │   ├── jetson.yaml            # Jetson overrides
│   │   └── rpi.yaml               # Raspberry Pi overrides
│   └── requirements/
│       ├── base.txt               # Common dependencies
│       ├── jetson.txt             # Jetson CUDA deps
│       └── rpi.txt                # RPi CPU deps
│
├── dashboard/                     # Dashboard code (Laptop)
│   ├── backend/                   # Flask API server
│   │   ├── app/
│   │   │   ├── routes/
│   │   │   │   └── detections.py  # Detection CRUD + stats
│   │   │   ├── models/
│   │   │   │   └── detection.py   # DB models with fullness
│   │   │   └── __init__.py        # Flask app factory
│   │   └── requirements.txt
│   └── frontend/                  # React dashboard
│       ├── src/
│       │   ├── components/
│       │   │   ├── DetectionFeed.jsx
│       │   │   ├── BinFullnessCard.jsx
│       │   │   ├── SummaryCard.jsx
│       │   │   ├── StatsPanel.jsx
│       │   │   ├── DetectionHistory.jsx
│       │   │   └── BoundingBoxOverlay.jsx
│       │   ├── services/
│       │   │   ├── api.js
│       │   │   └── socket.js
│       │   ├── App.jsx
│       │   └── styles.css
│       └── package.json
│
├── docker/                        # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── scripts/
│   ├── build_dashboard.sh         # Start dashboard stack
│   ├── build_edge.sh              # Run edge detection
│   └── setup_edge.sh              # Install edge dependencies
│
├── README.md                      # This file
└── CLAUDE.md                      # Development instructions
```

## Configuration

### Edge Device Configuration (`edge/config/config.yaml`)

```yaml
camera:
  device: "/dev/video0"
  width: 640
  height: 480
  fps: 5

model:
  name: "microsoft/Florence-2-base"
  device: "auto"  # auto, cuda, cpu
  confidence_threshold: 0.3

server:
  protocol: "rest"
  url: "http://YOUR_LAPTOP_IP:5000"  # <-- Change this!
  send_images: true
```

### Docker Compose Ports

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| MySQL | 3306 | 3303 |
| Backend | 5000 | 5000 |
| Frontend | 80 | 3004 |

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login with username/password, returns JWT token |
| GET | `/api/auth/verify` | Verify JWT token is valid |
| POST | `/api/auth/logout` | Logout (client should discard token) |

**Default credentials**: `admin` / `admin`

### Detection Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| POST | `/api/detections` | JWT + HMAC | Submit detection from edge device |
| GET | `/api/detections` | No | List detections (paginated) |
| GET | `/api/detections/latest` | No | Get most recent detection |
| GET | `/api/detections/<id>` | No | Get specific detection |
| GET | `/api/detections/<id>/image` | No | Get detection image |
| DELETE | `/api/detections/<id>` | No | Delete detection |
| GET | `/api/detections/stats` | No | Get statistics with fullness distribution |
| GET | `/api/detections/export` | No | Export detections as JSON |
| GET | `/api/health` | No | Health check |

### Security

The edge device authenticates with the dashboard using:
1. **JWT Token**: Obtained from `/api/auth/login`, sent in `Authorization: Bearer <token>` header
2. **HMAC-SHA256**: Request body signed and sent in `X-HMAC-Signature` header

### Detection Payload Format

```json
{
  "timestamp": "2025-12-02T10:30:00Z",
  "frame_id": "uuid",
  "detections": [
    {
      "label": "trash bin",
      "confidence": 0.95,
      "bbox": {"x": 100, "y": 150, "width": 200, "height": 300},
      "bin_fullness": "75-90%",
      "bin_fullness_percent": 82,
      "is_content": false
    },
    {
      "label": "plastic bottle",
      "confidence": 0.85,
      "bbox": {"x": 120, "y": 180, "width": 50, "height": 80},
      "is_content": true,
      "parent_bin_id": 0
    }
  ],
  "inference_time_ms": 45.2,
  "image_width": 640,
  "image_height": 480,
  "bin_detected": true,
  "bin_count": 1,
  "overall_fullness": "75-90%",
  "overall_fullness_percent": 82,
  "status_summary": "Containers: 1 (trash bin)\nFill Level: 82% [HIGH]\nAction: Schedule collection soon.",
  "image_base64": "..."
}
```

**Note**: Objects with `is_content: true` are items detected inside bins. They have a `parent_bin_id` indicating which bin they belong to.

## Platform Support

| Platform | GPU | Target FPS | Model |
|----------|-----|------------|-------|
| Jetson Orin | CUDA | 5-10 | Florence-2-base |
| Raspberry Pi 4 | CPU | 1-2 | Florence-2-base (quantized) |

## Development

### Running Backend Locally

```bash
cd dashboard/backend
pip install -r requirements.txt

# Start MySQL
docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=detections -p 3303:3306 mysql:8.0

# Run Flask
export DATABASE_URL="mysql+pymysql://root:root@localhost:3303/detections"
python run.py
```

### Running Frontend Locally

```bash
cd dashboard/frontend
npm install
npm run dev
```

## Troubleshooting

### Camera not detected
```bash
ls -la /dev/video*
./scripts/build_edge.sh test-camera
```

### Model loading fails
```bash
./scripts/build_edge.sh test-model
free -h
```

### dtype mismatch error
If you see "Input type (float) and bias type (c10::Half) should be the same":
- This is handled automatically by the detector
- Pixel values are cast to match model dtype (float16/float32)

### Connection to dashboard fails
1. Verify laptop IP in `edge/config/config.yaml`
2. Check firewall allows port 5000
3. Test: `curl http://LAPTOP_IP:5000/api/health`

## License

MIT License
