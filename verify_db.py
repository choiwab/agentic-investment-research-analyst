
from pymongo import MongoClient

uri = "mongodb://root:password@localhost:27017/"

client = MongoClient(uri)

db = client['test']

collections = db.list_collection_names()


print("Collections in the 'test' database:")
print(collections)
