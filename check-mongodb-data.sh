#!/bin/bash

# MongoDB Data Check Script
# Run this on your GCP instance to check data collection status

echo "======================================================"
echo "  MongoDB Data Collection Status Check"
echo "======================================================"
echo ""

# Get MongoDB password from .env
MONGO_PASSWORD=$(grep MONGO_INITDB_ROOT_PASSWORD .env | cut -d '=' -f2)

if [ -z "$MONGO_PASSWORD" ]; then
    echo "Warning: Could not find MONGO_INITDB_ROOT_PASSWORD in .env"
    MONGO_PASSWORD="password"
fi

echo "1. Checking MongoDB collections and document counts..."
echo "------------------------------------------------------"
docker exec -it mongo mongosh -u root -p "$MONGO_PASSWORD" --authenticationDatabase admin --quiet --eval "
use financial_data
print('Database: financial_data')
print('')
print('Collections and document counts:')
print('================================')
var collections = db.getCollectionNames();
collections.forEach(function(collName) {
  var count = db[collName].countDocuments();
  print(collName.padEnd(25) + ': ' + count.toString().padStart(8) + ' documents');
});
print('')
print('Total collections: ' + collections.length);
"

echo ""
echo "2. Checking Airflow DAG status..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml exec -T airflow-scheduler airflow dags list | grep financial_data_etl

echo ""
echo "3. Checking recent DAG runs..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml exec -T airflow-scheduler airflow dags list-runs -d financial_data_etl --limit 5

echo ""
echo "4. Checking for DAG errors..."
echo "------------------------------------------------------"
docker compose -f docker-compose.prod.yml logs airflow-scheduler | grep -i "error\|exception\|failed" | tail -20

echo ""
echo "5. Sample data from companies collection..."
echo "------------------------------------------------------"
docker exec -it mongo mongosh -u root -p "$MONGO_PASSWORD" --authenticationDatabase admin --quiet --eval "
use financial_data
var count = db.companies.countDocuments();
if (count > 0) {
  print('Found ' + count + ' companies');
  print('Sample company:');
  printjson(db.companies.findOne());
} else {
  print('No companies found yet.');
}
"

echo ""
echo "======================================================"
echo "  Check Complete"
echo "======================================================"
