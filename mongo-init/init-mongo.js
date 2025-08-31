db.createCollection('companies');
db.companies.createIndex({ ticker: 1 });

db.createCollection('earnings_reports');
db.earnings_reports.createIndex({ ticker: 1 });

db.createCollection('sec_fillings');
db.sec_fillings.createIndex({ ticker: 1 });

db.createCollection('news');
db.news.createIndex({ ticker: 1 });

db.createCollection('market_data');
db.market_data.createIndex({ ticker: 1 });
