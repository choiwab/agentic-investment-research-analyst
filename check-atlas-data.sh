#!/bin/bash

# MongoDB Atlas Data Check Script
# Run this on your GCP instance to verify data is being imported to Atlas

echo "======================================================"
echo "  MongoDB Atlas Data Import Verification"
echo "======================================================"
echo ""

# Get Atlas URI from .env
ATLAS_URI=$(grep ATLAS_URI .env | grep -v "^#" | cut -d '=' -f2)

if [ -z "$ATLAS_URI" ]; then
    echo "❌ ERROR: ATLAS_URI not found in .env file"
    echo ""
    echo "Please add ATLAS_URI to your .env file:"
    echo "ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
    echo ""
    exit 1
fi

echo "✓ ATLAS_URI found in .env"
echo ""

# Check if Airflow is using Atlas
echo "1. Checking if Airflow is configured to use Atlas..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml logs airflow-scheduler 2>/dev/null | grep -i "mongodb\|atlas" | tail -5

if docker compose -f docker-compose.prod.yml logs airflow-scheduler 2>/dev/null | grep -q "Using MongoDB Atlas"; then
    echo "✓ Airflow is using MongoDB Atlas"
else
    echo "⚠ Warning: Could not confirm Atlas usage in logs"
    echo "  This might be normal if no DAG has run yet"
fi
echo ""

# Check DAG status
echo "2. Checking Airflow DAG status..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml exec -T airflow-scheduler airflow dags list 2>/dev/null | grep financial_data_etl || echo "DAG not found or scheduler not ready"
echo ""

# Check recent DAG runs
echo "3. Recent DAG runs..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml exec -T airflow-scheduler airflow dags list-runs -d financial_data_etl --limit 5 2>/dev/null || echo "No runs found yet"
echo ""

# Try to connect to Atlas and check data
echo "4. Attempting to connect to Atlas and check data..."
echo "------------------------------------------------------"

# Check if mongosh is installed
if command -v mongosh &> /dev/null; then
    echo "Connecting to Atlas..."

    # Use mongosh to check collections and counts
    mongosh "$ATLAS_URI/financial_data" --quiet --eval "
    try {
        print('✓ Connected to MongoDB Atlas successfully!');
        print('');
        print('Database: financial_data');
        print('');
        print('Collections and document counts:');
        print('================================');

        var collections = db.getCollectionNames();
        if (collections.length === 0) {
            print('⚠ No collections found yet. The DAG may not have run successfully.');
            print('');
            print('Possible reasons:');
            print('1. DAG is not enabled in Airflow UI');
            print('2. DAG has not completed its first run yet');
            print('3. DAG encountered errors (check Airflow UI)');
        } else {
            var totalDocs = 0;
            collections.forEach(function(collName) {
                var count = db[collName].countDocuments();
                totalDocs += count;
                print(collName.padEnd(25) + ': ' + count.toString().padStart(8) + ' documents');
            });
            print('');
            print('Total collections: ' + collections.length);
            print('Total documents: ' + totalDocs);

            if (totalDocs > 0) {
                print('');
                print('✓ Data is being successfully imported to Atlas!');
            }
        }
    } catch (err) {
        print('❌ Error connecting to Atlas:');
        print(err.message);
        print('');
        print('Please verify:');
        print('1. ATLAS_URI is correct in .env');
        print('2. Network access is allowed (0.0.0.0/0) in Atlas');
        print('3. Database user credentials are correct');
    }
    " 2>&1
else
    echo "ℹ mongosh is not installed on this system"
    echo ""
    echo "To install mongosh and check Atlas directly:"
    echo "  curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg"
    echo "  echo 'deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y mongodb-mongosh"
    echo ""
    echo "For now, please check Atlas web UI:"
    echo "  https://cloud.mongodb.com/ → Database → Browse Collections"
fi

echo ""
echo "======================================================"
echo "  Next Steps"
echo "======================================================"
echo ""
echo "1. Check MongoDB Atlas Web UI:"
echo "   https://cloud.mongodb.com/"
echo "   → Database → Browse Collections"
echo ""
echo "2. Check Airflow Web UI:"
echo "   http://$(curl -s ifconfig.me):8080"
echo "   → Enable financial_data_etl DAG if not already enabled"
echo ""
echo "3. Trigger DAG manually to test:"
echo "   docker compose -f docker-compose.prod.yml exec airflow-scheduler airflow dags trigger financial_data_etl"
echo ""
echo "======================================================"
