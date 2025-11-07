#!/bin/bash

# Oracle Cloud Instance Setup Script for Airflow ETL
# Run this script on your Oracle Cloud ARM instance

set -e  # Exit on error

echo "================================================"
echo "  Airflow ETL - Oracle Cloud Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running as ubuntu user
if [ "$USER" != "ubuntu" ]; then
    print_error "Please run this script as ubuntu user"
    exit 1
fi

# Update system
print_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
print_success "System updated"

# Install Docker
print_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
    print_success "Docker installed"
else
    print_success "Docker already installed"
fi

# Install Docker Compose
print_info "Installing Docker Compose..."
if ! command -v docker compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
    print_success "Docker Compose installed"
else
    print_success "Docker Compose already installed"
fi

# Install Git
print_info "Installing Git..."
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git
    print_success "Git installed"
else
    print_success "Git already installed"
fi

# Install useful tools
print_info "Installing additional tools..."
sudo apt-get install -y htop nano curl wget
print_success "Additional tools installed"

# Configure UFW Firewall
print_info "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 8080/tcp comment 'Airflow Web UI'
print_success "Firewall configured"

# Create necessary directories
print_info "Creating directories..."
mkdir -p ~/airflow-etl/airflow/logs
print_success "Directories created"

# Check if we're already in the repo
if [ -d ".git" ]; then
    print_info "Already in git repository"
else
    print_info "Repository needs to be cloned manually"
    echo ""
    echo "Next steps:"
    echo "1. Clone your repository: git clone https://github.com/choiwab/agentic-investment-research-analyst.git"
    echo "2. cd agentic-investment-research-analyst"
    echo "3. Create .env file with your configuration"
    echo "4. Run: docker compose -f docker-compose.prod.yml up -d"
fi

# Display version info
echo ""
print_info "Installation complete! Versions:"
echo "  Docker: $(docker --version)"
echo "  Docker Compose: $(docker compose version)"
echo "  Git: $(git --version)"
echo ""

# Check if need to re-login for docker group
if ! groups | grep -q docker; then
    print_info "Docker group added. Please log out and log back in for changes to take effect."
    echo ""
    echo "Run: exit"
    echo "Then SSH back in and continue with deployment"
else
    print_success "Docker group already active"
fi

echo ""
print_success "Setup complete!"
echo ""
echo "================================================"
echo "  Next Steps:"
echo "================================================"
echo "1. Log out and log back in (if docker group was just added)"
echo "2. Clone your repository (if not already done)"
echo "3. Create .env file with your credentials"
echo "4. Run: docker compose -f docker-compose.prod.yml up -d"
echo "5. Access Airflow UI at: http://$(curl -s ifconfig.me):8080"
echo ""
