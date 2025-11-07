# Oracle Cloud Deployment Guide

## Prerequisites Completed
- [x] Oracle Cloud account created
- [x] ARM instance created (VM.Standard.A1.Flex - 4 CPU, 24GB RAM)
- [x] SSH keys downloaded

## Step 2: Configure Firewall Rules

### 2.1 Oracle Cloud Console
1. Go to your instance details page
2. Click on the **Subnet** link
3. Click on the **Default Security List**
4. Click **"Add Ingress Rules"**
5. Add these rules:

**Rule 1 - Airflow Web UI:**
- Source CIDR: `0.0.0.0/0`
- Destination Port: `8080`
- Description: `Airflow Web UI`

**Rule 2 - MongoDB (optional, only if you need external access):**
- Source CIDR: `0.0.0.0/0`
- Destination Port: `27017`
- Description: `MongoDB`

### 2.2 Ubuntu Firewall (UFW)
Run these commands on your instance after SSH:

```bash
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8080/tcp    # Airflow
sudo ufw enable
```

## Step 3: Connect to Your Instance

```bash
# Replace <PATH_TO_KEY> with your downloaded private key path
# Replace <PUBLIC_IP> with your instance's public IP

chmod 600 <PATH_TO_KEY>
ssh -i <PATH_TO_KEY> ubuntu@<PUBLIC_IP>
```

## Step 4: Install Docker & Docker Compose

Once connected via SSH, run these commands:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt-get install -y docker-compose-plugin

# Verify installation
docker --version
docker compose version

# Log out and log back in for group changes to take effect
exit
```

Then SSH back in again.

## Step 5: Clone Your Repository

```bash
# Install git if not present
sudo apt-get install -y git

# Clone your repository
git clone https://github.com/choiwab/agentic-investment-research-analyst.git
cd agentic-investment-research-analyst
```

## Step 6: Set Up Environment Variables

```bash
# Create .env file
nano .env
```

Add your environment variables (copy from your local .env):
```
FINNHUB_API_KEY=your_api_key_here
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=password
MONGO_CONNECTION_STRING=mongodb://root:password@mongo:27017
AIRFLOW_UID=50000
```

Save and exit (Ctrl+X, then Y, then Enter)

## Step 7: Start the Services

```bash
# Create necessary directories
mkdir -p airflow/logs

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

## Step 8: Access Airflow Web UI

1. Open browser: `http://<YOUR_PUBLIC_IP>:8080`
2. Login credentials:
   - Username: `airflow`
   - Password: `airflow`

## Step 9: Enable and Start Your DAG

1. In Airflow UI, find `financial_data_etl` DAG
2. Toggle the switch to **ON** (enable)
3. Click **"Trigger DAG"** to start immediately
4. Monitor the runs in the UI

## Useful Commands

```bash
# View all containers
docker compose ps

# View logs for specific service
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver

# Restart services
docker compose restart

# Stop services
docker compose down

# Update code and restart
git pull
docker compose down
docker compose build
docker compose up -d

# Check MongoDB data
docker exec -it mongo mongosh -u root -p password --authenticationDatabase admin
# Then in mongo shell:
use financial_data
db.companies.countDocuments()
db.news.countDocuments()
exit
```

## Monitoring

```bash
# Check resource usage
docker stats

# Check disk space
df -h

# Check system resources
htop  # (install with: sudo apt-get install htop)
```

## Troubleshooting

### If containers won't start:
```bash
# Check logs
docker compose logs

# Restart everything
docker compose down -v  # WARNING: This removes volumes/data
docker compose up -d
```

### If port 8080 is not accessible:
1. Check Oracle Cloud security list rules
2. Check UFW: `sudo ufw status`
3. Check container: `docker compose ps`

### If running out of memory:
```bash
# Check memory usage
free -h

# Check which containers use most memory
docker stats
```

## Auto-restart on Reboot

To ensure services start automatically after system reboot:

```bash
# Edit crontab
crontab -e

# Add this line:
@reboot cd /home/ubuntu/agentic-investment-research-analyst && docker compose up -d
```

## Cost Monitoring

This setup is **100% free** as long as you stay within:
- 4 OCPUs ARM
- 24 GB RAM
- 200 GB storage
- 10 TB outbound data transfer/month

Check your usage in Oracle Cloud Console > Billing & Cost Management
