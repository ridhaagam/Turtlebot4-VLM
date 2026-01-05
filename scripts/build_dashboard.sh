#!/bin/bash
# =============================================================================
# Build and run the Detection Dashboard on your laptop
# ONE COMMAND to start everything: MySQL + Flask Backend + React Frontend
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Detection Dashboard Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create .env file if not exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file with default passwords...${NC}"
    cat > .env << EOF
# Database Configuration
DB_ROOT_PASSWORD=rootpassword123
DB_PASSWORD=adminpassword123

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "change-this-secret-key-in-production")

# Optional: Change these for production
# FLASK_ENV=development
EOF
    echo -e "${YELLOW}WARNING: Using default passwords. Change in .env for production!${NC}"
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Determine docker compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Parse arguments
ACTION=${1:-up}

case $ACTION in
    up|start)
        echo -e "${GREEN}Building and starting services...${NC}"
        $COMPOSE_CMD -f docker/docker-compose.yml up --build -d

        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Dashboard is starting up!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "Frontend:  ${YELLOW}http://localhost:3004${NC}"
        echo -e "API:       ${YELLOW}http://localhost:5000${NC}"
        echo -e "MySQL:     ${YELLOW}localhost:3303${NC}"
        echo ""
        echo -e "View logs: ${YELLOW}$0 logs${NC}"
        echo -e "Stop:      ${YELLOW}$0 down${NC}"
        echo ""

        # Wait for services to be ready
        echo "Waiting for services to be ready..."
        sleep 5

        # Check health
        if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}Backend is healthy!${NC}"
        else
            echo -e "${YELLOW}Backend is still starting...${NC}"
        fi
        ;;

    down|stop)
        echo -e "${YELLOW}Stopping services...${NC}"
        $COMPOSE_CMD -f docker/docker-compose.yml down
        echo -e "${GREEN}Services stopped.${NC}"
        ;;

    restart)
        echo -e "${YELLOW}Restarting services...${NC}"
        $COMPOSE_CMD -f docker/docker-compose.yml restart
        echo -e "${GREEN}Services restarted.${NC}"
        ;;

    logs)
        $COMPOSE_CMD -f docker/docker-compose.yml logs -f
        ;;

    clean)
        echo -e "${RED}WARNING: This will delete all data including the database!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $COMPOSE_CMD -f docker/docker-compose.yml down -v
            echo -e "${GREEN}All data cleaned.${NC}"
        fi
        ;;

    rebuild)
        echo -e "${YELLOW}Rebuilding with fresh database (deletes all data)...${NC}"
        $COMPOSE_CMD -f docker/docker-compose.yml down -v
        $COMPOSE_CMD -f docker/docker-compose.yml up --build -d
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Dashboard rebuilt with fresh database!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "Frontend:  ${YELLOW}http://localhost:3004${NC}"
        echo -e "API:       ${YELLOW}http://localhost:5000${NC}"
        echo -e "MySQL:     ${YELLOW}localhost:3303${NC}"
        echo ""
        ;;

    status)
        $COMPOSE_CMD -f docker/docker-compose.yml ps
        ;;

    *)
        echo "Usage: $0 {up|down|restart|logs|clean|rebuild|status}"
        echo ""
        echo "Commands:"
        echo "  up, start   - Build and start all services"
        echo "  down, stop  - Stop all services"
        echo "  restart     - Restart all services"
        echo "  logs        - View logs (follow mode)"
        echo "  clean       - Stop and remove all data (WARNING: destructive)"
        echo "  rebuild     - Full rebuild with fresh database (use after schema changes)"
        echo "  status      - Show service status"
        exit 1
        ;;
esac
