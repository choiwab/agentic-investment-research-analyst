# Quick Start - Oracle Cloud Deployment

## What You Need

1. **Oracle Cloud Account** - Sign up at https://www.oracle.com/cloud/free/
2. **Your Finnhub API Key** - Get it from https://finnhub.io/
3. **SSH Client** - Terminal (Mac/Linux) or PuTTY (Windows)

## Step-by-Step Instructions

### 1. Create Oracle Cloud VM (5 minutes)

1. Go to https://cloud.oracle.com/
2. Navigate to: **Compute → Instances → Create Instance**
3. Configure:
   - Name: `airflow-etl`
   - Image: **Ubuntu 22.04**
   - Shape: Click "Change Shape" → Select **Ampere (ARM)** → **VM.Standard.A1.Flex**
   - CPUs: **4 OCPU**
   - Memory: **24 GB**
   - Network: Use default VCN
   - SSH Keys: **Download the private key!** (save it somewhere safe)
4. Click **Create**
5. **Note down the Public IP address** (shows after ~2 min)

### 2. Open Firewall Ports (2 minutes)

In Oracle Cloud Console:
1. Click on your instance
2. Click the **Subnet** link
3. Click **Default Security List**
4. Click **Add Ingress Rules**
5. Add:
   - Source: `0.0.0.0/0`
   - Destination Port: `8080`
   - Description: `Airflow UI`
6. Click **Add Ingress Rules**

### 3. Connect to Your Instance (1 minute)

```bash
# Make the SSH key readable only by you
chmod 600 /path/to/downloaded-private-key

# Connect (replace IP with yours)
ssh -i /path/to/downloaded-private-key ubuntu@YOUR_PUBLIC_IP
```

### 4. Run Setup Script (5 minutes)

Once connected to your instance:

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/choiwab/agentic-investment-research-analyst/main/oracle-setup.sh -o setup.sh
chmod +x setup.sh
./setup.sh

# Log out and log back in for docker group to take effect
exit
```

Then SSH back in:
```bash
ssh -i /path/to/downloaded-private-key ubuntu@YOUR_PUBLIC_IP
```

### 5. Clone Repository & Configure (3 minutes)

```bash
# Clone the repository
git clone https://github.com/choiwab/agentic-investment-research-analyst.git
cd agentic-investment-research-analyst

# Checkout the right branch (if needed)
git checkout fix/airflow-etl-fix

# Create .env file
nano .env
```

Paste this configuration (replace with your actual API key):
```
# Airflow Configuration
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow123

# MongoDB Configuration
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=securePassword123
MONGO_CONNECTION_STRING=mongodb://root:securePassword123@mongo:27017

# Finnhub API
FINNHUB_API_KEY=your_actual_finnhub_api_key_here
```

Save: `Ctrl+X`, then `Y`, then `Enter`

### 6. Start the Services (3 minutes)

```bash
# Create logs directory
mkdir -p airflow/logs

# Start everything
docker compose -f docker-compose.prod.yml up -d

# Wait ~2 minutes, then check status
docker compose -f docker-compose.prod.yml ps
```

All services should show "Up" or "healthy"

### 7. Access Airflow UI & Start DAG (2 minutes)

1. Open browser: `http://YOUR_PUBLIC_IP:8080`
2. Login:
   - Username: `airflow`
   - Password: `airflow123` (or what you set)
3. Find `financial_data_etl` DAG
4. Toggle the switch to **ON** (enable)
5. Click **"Trigger DAG"** or wait for auto-start

### 8. Monitor Your DAG

View logs in real-time:
```bash
docker compose -f docker-compose.prod.yml logs -f airflow-scheduler
```

Check MongoDB data:
```bash
docker exec -it mongo mongosh -u root -p securePassword123 --authenticationDatabase admin
```
Then in MongoDB shell:
```javascript
use financial_data
db.companies.countDocuments()
db.news.countDocuments()
db.market_data.countDocuments()
exit
```

## Useful Commands

```bash
# View all containers
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Restart services
docker compose -f docker-compose.prod.yml restart

# Stop everything
docker compose -f docker-compose.prod.yml down

# Update code and restart
git pull
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Check system resources
docker stats
htop
df -h
```

## Auto-Start on Reboot

```bash
crontab -e
# Add this line:
@reboot cd /home/ubuntu/agentic-investment-research-analyst && docker compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

**Can't access Airflow UI?**
- Check Oracle Cloud security list has port 8080 open
- Check UFW: `sudo ufw status`
- Check containers: `docker compose -f docker-compose.prod.yml ps`

**Services won't start?**
```bash
# Check logs for errors
docker compose -f docker-compose.prod.yml logs

# Restart
docker compose -f docker-compose.prod.yml restart
```

**Out of memory?**
```bash
# Check usage
free -h
docker stats

# Reduce batch sizes in etl/dags/financial_data_dag.py if needed
```

## Cost

This is **100% FREE** - Oracle's always-free tier includes:
- ARM VM (4 CPU, 24GB RAM)
- 200 GB storage
- 10 TB outbound transfer/month

## Support

For detailed instructions, see [ORACLE_DEPLOYMENT.md](ORACLE_DEPLOYMENT.md)
