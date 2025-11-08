# Google Cloud Platform (GCP) Deployment Guide

## GCP Free Tier Benefits

**New Users Get:**
- **$300 free credit** (valid for 90 days)
- After credits expire, **Always Free tier** includes:
  - 1 f1-micro VM instance per month (US regions only)
  - 30 GB standard persistent disk
  - 1 GB network egress per month

**What You Can Run:**
- f1-micro: 0.6 GB RAM, 1 shared vCPU (bursting)
- Enough for lightweight Airflow + MongoDB (with optimization)
- OR use e2-medium during free trial ($300 credit): 2 vCPU, 4GB RAM

## Step 1: Create GCP Account

1. Go to https://cloud.google.com/free
2. Click **"Get started for free"**
3. Sign in with Google account
4. Enter billing information (won't be charged during trial)
5. Accept terms and click **"Start my free trial"**
6. You'll get **$300 credit** valid for 90 days

## Step 2: Create VM Instance

### 2.1 Open Google Cloud Console

1. Go to https://console.cloud.google.com/
2. Navigate to: **Menu (≡) → Compute Engine → VM instances**
3. Click **"Enable Compute Engine API"** (if prompted, wait ~2 minutes)
4. Click **"Create Instance"**

### 2.2 Configure Your VM

**Basic Configuration:**
- **Name**: `airflow-etl-server`
- **Region**: `us-central1` (Iowa) - eligible for free tier
- **Zone**: `us-central1-a` (or any zone)

**Machine Configuration:**

**Option A: During Free Trial ($300 credit) - RECOMMENDED**
- **Series**: E2
- **Machine type**: `e2-medium` (2 vCPU, 4 GB RAM)
- Cost: ~$25/month (covered by $300 credit for ~12 months)

**Option B: Always Free Tier (after credits expire)**
- **Series**: E2
- **Machine type**: `e2-micro` (2 vCPU, 1 GB RAM - shared core)
- Note: This is tight for Airflow. You may need to optimize.

**Boot Disk:**
- Click **"Change"**
- **Operating System**: Ubuntu
- **Version**: Ubuntu 22.04 LTS
- **Boot disk type**: Standard persistent disk
- **Size**: 30 GB (max for free tier)
- Click **"Select"**

**Firewall:**
- ✅ Check **"Allow HTTP traffic"**
- ✅ Check **"Allow HTTPS traffic"**

Click **"Create"** (takes ~1 minute)

## Step 3: Configure Firewall Rules

### 3.1 Create Firewall Rule for Airflow

1. Go to: **Menu (≡) → VPC network → Firewall**
2. Click **"Create Firewall Rule"**
3. Configure:
   - **Name**: `allow-airflow-ui`
   - **Logs**: Off
   - **Network**: default
   - **Priority**: 1000
   - **Direction**: Ingress
   - **Action on match**: Allow
   - **Targets**: All instances in the network
   - **Source filter**: IPv4 ranges
   - **Source IPv4 ranges**: `0.0.0.0/0`
   - **Protocols and ports**:
     - ✅ Specified protocols and ports
     - **TCP**: `8080`
4. Click **"Create"**

## Step 4: Connect to Your Instance

### 4.1 Connect via Browser (Easiest)

1. Go to **Compute Engine → VM instances**
2. Find your instance
3. Click **"SSH"** button
4. A browser window will open with terminal access

### 4.2 Connect via Local Terminal (Alternative)

```bash
# Install gcloud CLI first: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# SSH to instance
gcloud compute ssh airflow-etl-server --zone=us-central1-a
```

## Step 5: Install Docker & Docker Compose

Once connected via SSH, run these commands:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get install -y docker-compose-plugin

# Install useful tools
sudo apt-get install -y git htop nano

# Verify installation
docker --version
docker compose version

# IMPORTANT: Log out and log back in for group changes
exit
```

**Then reconnect:**
- Click **SSH** button again in GCP Console
- OR run: `gcloud compute ssh airflow-etl-server --zone=us-central1-a`

## Step 6: Clone Repository & Configure

```bash
# Clone the repository
git clone https://github.com/choiwab/agentic-investment-research-analyst.git
cd agentic-investment-research-analyst

# Checkout the correct branch
git checkout fix/airflow-etl-fix

# Create .env file
nano .env
```

**Paste this configuration** (replace with your actual values):

```bash
# Airflow Configuration
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=your_secure_password_here

# MongoDB Configuration
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=your_mongo_password_here
MONGO_CONNECTION_STRING=mongodb://root:your_mongo_password_here@mongo:27017

# Finnhub API
FINNHUB_API_KEY=your_finnhub_api_key_here

# OpenAI & Other Services (optional)
OPENAI_API_KEY=your_openai_api_key_here
FASTAPI_BASE_URL=http://localhost:8000
TAVILY_API_KEY=your_tavily_api_key_here
```

**Save the file:**
- Press `Ctrl+X`
- Press `Y`
- Press `Enter`

## Step 7: Deploy the Application

### 7.1 Create Necessary Directories

```bash
mkdir -p airflow/logs
```

### 7.2 Start Services

```bash
# Start all services in background
docker compose -f docker-compose.prod.yml up -d

# Wait ~2-3 minutes for initialization
# Check status
docker compose -f docker-compose.prod.yml ps
```

**Expected output:** All services should show "Up" or "healthy"

### 7.3 Monitor Startup

```bash
# Watch logs
docker compose -f docker-compose.prod.yml logs -f

# Press Ctrl+C to stop watching (services keep running)
```

## Step 8: Get Your VM's Public IP

```bash
# Get external IP
curl ifconfig.me
```

OR find it in GCP Console: **Compute Engine → VM instances** (External IP column)

## Step 9: Access Airflow Web UI

1. Open browser: `http://YOUR_EXTERNAL_IP:8080`
2. Login:
   - **Username**: `airflow`
   - **Password**: (what you set in .env)
3. Find `financial_data_etl` DAG
4. Click the **toggle switch** to enable it
5. Click **"Trigger DAG"** to start immediately

## Step 10: Monitor Your DAG

### View Logs

```bash
# Scheduler logs (shows DAG execution)
docker compose -f docker-compose.prod.yml logs -f airflow-scheduler

# Webserver logs
docker compose -f docker-compose.prod.yml logs -f airflow-webserver
```

### Check MongoDB Data

```bash
# Connect to MongoDB
docker exec -it mongo mongosh -u root -p your_mongo_password_here --authenticationDatabase admin

# In MongoDB shell:
use financial_data
db.companies.countDocuments()
db.news.countDocuments()
db.market_data.countDocuments()
db.earnings_reports.countDocuments()
exit
```

## Useful Commands

### Container Management

```bash
# View all containers
docker compose -f docker-compose.prod.yml ps

# View logs for all services
docker compose -f docker-compose.prod.yml logs -f

# Restart all services
docker compose -f docker-compose.prod.yml restart

# Stop all services
docker compose -f docker-compose.prod.yml down

# Stop and remove volumes (WARNING: deletes all data)
docker compose -f docker-compose.prod.yml down -v
```

### Update Code

```bash
cd ~/agentic-investment-research-analyst
git pull
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

### System Monitoring

```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check container resource usage
docker stats

# System monitor (interactive)
htop
```

## Auto-Start on VM Reboot

```bash
# Edit crontab
crontab -e

# Choose nano as editor (option 1)
# Add this line at the end:
@reboot sleep 30 && cd /home/$USER/agentic-investment-research-analyst && docker compose -f docker-compose.prod.yml up -d

# Save: Ctrl+X, Y, Enter
```

## Optimization for Low Memory (f1-micro/e2-micro)

If using the always-free tier (1GB RAM), you'll need to optimize:

### 1. Reduce Airflow Resource Usage

Create `docker-compose.gcp-micro.yml` based on prod version with:
- Reduced batch sizes
- Lower parallel workers
- Memory limits on containers

### 2. Use MongoDB Atlas (Free Tier)

Instead of running MongoDB locally:
- Sign up at https://www.mongodb.com/cloud/atlas
- Create free cluster (512MB)
- Update `MONGO_CONNECTION_STRING` in .env

### 3. Adjust DAG Schedule

Edit `etl/dags/financial_data_dag.py`:
```python
schedule_interval=timedelta(hours=6)  # Instead of @continuous
```

## Cost Estimates

### During Free Trial ($300 credit):
- **e2-medium** (2 vCPU, 4GB): ~$25/month → **12 months covered**
- Network egress: Minimal (mostly API calls)
- Storage: Free (30GB)

### After Free Trial (Always Free Tier):
- **e2-micro** (1 GB RAM): **$0/month** (always free in us-central1/us-west1/us-east1)
- Network egress: 1GB/month free
- Storage: 30GB free

### Recommendation:
Use **e2-medium** during trial for best performance, then:
- Switch to e2-micro + MongoDB Atlas free tier
- OR keep e2-medium for ~$25/month

## Troubleshooting

### Can't Access Airflow UI

```bash
# Check firewall rules
gcloud compute firewall-rules list

# Check if containers are running
docker compose -f docker-compose.prod.yml ps

# Check if port 8080 is listening
sudo netstat -tlnp | grep 8080
```

### Out of Memory Errors

```bash
# Check memory usage
free -h
docker stats

# Reduce batch sizes in DAG
nano etl/dags/financial_data_dag.py
# Change: batch_size: 50 → 20
# Change: max_workers: 5 → 2

# Restart
docker compose -f docker-compose.prod.yml restart
```

### Services Won't Start

```bash
# Check logs for errors
docker compose -f docker-compose.prod.yml logs

# Try rebuilding
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build --no-cache
docker compose -f docker-compose.prod.yml up -d
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Clean up logs
cd ~/agentic-investment-research-analyst
rm -rf airflow/logs/*
```

## Security Best Practices

1. **Change default passwords** in .env
2. **Restrict Airflow access** - Update firewall rule to only your IP:
   ```bash
   # Get your IP
   curl ifconfig.me

   # Update firewall rule source to: YOUR_IP/32
   ```
3. **Keep system updated**:
   ```bash
   sudo apt-get update && sudo apt-get upgrade -y
   ```
4. **Monitor billing** in GCP Console

## Stopping Your VM (To Save Credits)

```bash
# Stop VM when not in use
gcloud compute instances stop airflow-etl-server --zone=us-central1-a

# Start it again
gcloud compute instances start airflow-etl-server --zone=us-central1-a
```

Note: You're not charged when VM is stopped, but disk storage still counts.

## Support & Resources

- **GCP Free Tier**: https://cloud.google.com/free
- **GCP Documentation**: https://cloud.google.com/docs
- **Airflow Documentation**: https://airflow.apache.org/docs/
- **Project Repository**: https://github.com/choiwab/agentic-investment-research-analyst

## Next Steps

1. Monitor your DAG runs in Airflow UI
2. Check MongoDB data collection
3. Set up monitoring/alerting (optional)
4. Consider MongoDB Atlas for better reliability
5. Optimize batch sizes based on API rate limits
