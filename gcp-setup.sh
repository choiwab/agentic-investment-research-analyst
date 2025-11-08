#!/bin/bash

# GCP Instance Setup Script for Airflow ETL
# Run this script on your Google Cloud VM instance

set -e  # Exit on error

echo "================================================"
echo "  Airflow ETL - GCP Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_step() {
    echo -e "${BLUE}➜ $1${NC}"
}

# Update system
print_step "Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
print_success "System updated"

# Install Docker
print_step "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh > /dev/null 2>&1
    sudo usermod -aG docker $USER
    rm get-docker.sh
    print_success "Docker installed"
else
    print_success "Docker already installed"
fi

# Install Docker Compose
print_step "Installing Docker Compose..."
if ! command -v docker compose &> /dev/null; then
    sudo apt-get install -y -qq docker-compose-plugin
    print_success "Docker Compose installed"
else
    print_success "Docker Compose already installed"
fi

# Install Git
print_step "Installing Git..."
if ! command -v git &> /dev/null; then
    sudo apt-get install -y -qq git
    print_success "Git installed"
else
    print_success "Git already installed"
fi

# Install useful tools
print_step "Installing additional tools..."
sudo apt-get install -y -qq htop nano curl wget net-tools
print_success "Additional tools installed"

# Create necessary directories
print_step "Creating directories..."
mkdir -p ~/airflow-etl
print_success "Directories created"

# Display system info
echo ""
print_info "System Information:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Disk space: $(df -h / | awk 'NR==2 {print $4}') available"
echo ""

# Display version info
print_info "Installation complete! Versions:"
echo "  Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
echo "  Docker Compose: $(docker compose version | cut -d' ' -f4)"
echo "  Git: $(git --version | cut -d' ' -f3)"
echo ""

# Check if need to re-login for docker group
if ! groups | grep -q docker; then
    print_info "Docker group added. You need to reconnect for changes to take effect."
    echo ""
    echo -e "${YELLOW}IMPORTANT: Close this SSH session and reconnect${NC}"
    echo ""
    echo "After reconnecting, run these commands:"
    echo "  1. git clone https://github.com/choiwab/agentic-investment-research-analyst.git"
    echo "  2. cd agentic-investment-research-analyst"
    echo "  3. git checkout fix/airflow-etl-fix"
    echo "  4. nano .env  # Create your environment file"
    echo "  5. mkdir -p airflow/logs"
    echo "  6. docker compose -f docker-compose.prod.yml up -d"
else
    print_success "Docker group already active"
    echo ""
    print_step "Next Steps:"
    echo "  1. git clone https://github.com/choiwab/agentic-investment-research-analyst.git"
    echo "  2. cd agentic-investment-research-analyst"
    echo "  3. git checkout fix/airflow-etl-fix"
    echo "  4. nano .env  # Create your environment file"
    echo "  5. mkdir -p airflow/logs"
    echo "  6. docker compose -f docker-compose.prod.yml up -d"
fi

# Get external IP
echo ""
EXTERNAL_IP=$(curl -s ifconfig.me || echo "Unable to determine")
if [ "$EXTERNAL_IP" != "Unable to determine" ]; then
    print_info "Your VM's External IP: ${EXTERNAL_IP}"
    echo "  After deployment, access Airflow at: http://${EXTERNAL_IP}:8080"
fi

echo ""
print_success "Setup complete!"
echo ""
echo "================================================"
