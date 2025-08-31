
from pymongo import MongoClient

uri = "mongodb://root:password@localhost:27017/"

client = MongoClient(uri)

db = client['test']

print("Collections and Indexes in database:\n")

for collection_name in db.list_collection_names():
    print(f"Collection: {collection_name}")
    
    indexes = db[collection_name].list_indexes()
    for idx in indexes:
        print(f"  ➤ Index: {idx}")
    print()