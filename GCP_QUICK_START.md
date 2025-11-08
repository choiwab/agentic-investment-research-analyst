# Quick Start - Google Cloud Platform Deployment

## What You Get

✅ **$300 free credit** (90 days)
✅ **e2-medium VM**: 2 vCPU, 4GB RAM (~12 months with credit)
✅ **After credit**: e2-micro always free (1GB RAM)
✅ **30GB storage** (always free)
✅ **24/7 operation**

## Total Time: ~20 Minutes

---

## Step 1: Create GCP Account (5 min)

1. Go to https://cloud.google.com/free
2. Click **"Get started for free"**
3. Sign in with Google account
4. Enter payment info (won't charge during trial)
5. Get **$300 free credit**!

---

## Step 2: Create VM Instance (3 min)

1. Go to https://console.cloud.google.com/
2. Click **Menu (≡) → Compute Engine → VM instances**
3. Click **"Enable Compute Engine API"** (wait ~2 min)
4. Click **"Create Instance"**

**Configure:**
- **Name**: `airflow-etl-server`
- **Region**: `us-central1` (Iowa)
- **Zone**: `us-central1-a`
- **Machine type**:
  - Click "CHANGE"
  - Select **E2**
  - Choose **e2-medium** (2 vCPU, 4 GB)
- **Boot disk**: Click "CHANGE"
  - OS: **Ubuntu**
  - Version: **Ubuntu 22.04 LTS**
  - Size: **30 GB**
  - Click "SELECT"
- **Firewall**:
  - ✅ Allow HTTP traffic
  - ✅ Allow HTTPS traffic

5. Click **"CREATE"**

**Note your External IP** (shows in instances list)

---

## Step 3: Add Firewall Rule (2 min)

1. Go to **Menu (≡) → VPC network → Firewall**
2. Click **"Create Firewall Rule"**
3. Configure:
   - **Name**: `allow-airflow-ui`
   - **Direction**: Ingress
   - **Action**: Allow
   - **Targets**: All instances in the network
   - **Source IPv4 ranges**: `0.0.0.0/0`
   - **TCP ports**: `8080`
4. Click **"CREATE"**

---

## Step 4: Connect & Install Docker (5 min)

1. Go to **Compute Engine → VM instances**
2. Click **"SSH"** button next to your instance
3. In the SSH terminal, run:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose & tools
sudo apt-get update
sudo apt-get install -y docker-compose-plugin git htop nano

# Exit and reconnect
exit
```

4. Click **"SSH"** button again to reconnect

---

## Step 5: Clone & Configure (3 min)

```bash
# Clone repository
git clone https://github.com/choiwab/agentic-investment-research-analyst.git
cd agentic-investment-research-analyst
git checkout fix/airflow-etl-fix

# Create environment file
nano .env
```

**Paste this** (replace YOUR_API_KEY):

```
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow123

MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=mongoPass123
MONGO_CONNECTION_STRING=mongodb://root:mongoPass123@mongo:27017

FINNHUB_API_KEY=YOUR_API_KEY_HERE
```

**Save**: `Ctrl+X` → `Y` → `Enter`

---

## Step 6: Start Services (2 min)

```bash
# Create logs directory
mkdir -p airflow/logs

# Start everything
docker compose -f docker-compose.prod.yml up -d

# Wait 2-3 minutes, then check
docker compose -f docker-compose.prod.yml ps
```

All services should show "Up" or "healthy"

---

## Step 7: Access Airflow & Start DAG (2 min)

1. **Get your VM's IP**:
   ```bash
   curl ifconfig.me
   ```

2. **Open browser**: `http://YOUR_IP:8080`

3. **Login**:
   - Username: `airflow`
   - Password: `airflow123`

4. **Enable DAG**:
   - Find `financial_data_etl`
   - Toggle switch to **ON**
   - Click **"Trigger DAG"**

---

## 🎉 Done! Your ETL is Running 24/7

### Monitor Your Data

```bash
# View DAG logs
docker compose -f docker-compose.prod.yml logs -f airflow-scheduler

# Check MongoDB data
docker exec -it mongo mongosh -u root -p mongoPass123 --authenticationDatabase admin

# In MongoDB shell:
use financial_data
db.companies.countDocuments()
db.news.countDocuments()
exit
```

### Useful Commands

```bash
# View all containers
docker compose -f docker-compose.prod.yml ps

# Restart services
docker compose -f docker-compose.prod.yml restart

# Stop services
docker compose -f docker-compose.prod.yml down

# Update code
git pull
docker compose -f docker-compose.prod.yml restart
```

### Auto-Start on Reboot

```bash
crontab -e
# Add this line:
@reboot sleep 30 && cd $HOME/agentic-investment-research-analyst && docker compose -f docker-compose.prod.yml up -d
```

---

## Cost Breakdown

### During Free Trial (90 days):
- **$300 credit** → covers ~12 months of e2-medium
- **e2-medium**: ~$25/month
- **Storage**: Free (30GB)

### After Free Trial:
- **e2-micro**: **$0/month** (always free!)
- Requires optimization (lower batch sizes)
- OR continue with e2-medium for ~$25/month

---

## Troubleshooting

**Can't access Airflow UI?**
```bash
# Check firewall
gcloud compute firewall-rules list | grep 8080

# Check containers
docker compose -f docker-compose.prod.yml ps
```

**Out of memory?**
- Edit `etl/dags/financial_data_dag.py`
- Reduce `batch_size: 50` → `20`
- Reduce `max_workers: 5` → `2`

**Services won't start?**
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs

# Rebuild
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

---

## Need More Help?

See detailed guide: [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md)

---

## Your DAG Will:

✅ Run **continuously** collecting data
✅ Process **50 basic tickers** per run
✅ Process **20 detailed tickers** per run
✅ Use **parallel processing** (5+3 workers)
✅ **Auto-restart** on failures
✅ Collect data **24/7** even when your laptop is off!

Enjoy your always-on financial data pipeline! 📊🚀
