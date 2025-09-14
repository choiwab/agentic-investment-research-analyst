from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

ATLAS_URI = os.getenv("ATLAS_URI")
client = MongoClient(ATLAS_URI)
db = client["test"]

app = FastAPI()

@app.get("/companies")
def get_companies():
    # Project _id as ticker in the response
    companies = []
    for doc in db.companies.find({}, {}):
        company = dict(doc)
        company["ticker"] = company.pop("_id", None)
        companies.append(company)
    return {"companies": companies}

@app.get("/company/{ticker}")
def get_company(ticker: str):
    company = db.companies.find_one({"_id": ticker}, {"_id": 0})
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company

@app.post("/company/")
def add_company(company: dict):
    if "_id" not in company:
        raise HTTPException(status_code=400, detail="Company must have a ticker (_id)")
    if db.companies.find_one({"_id": company["_id"]}):
        raise HTTPException(status_code=400, detail="Company with this ticker already exists")
    db.companies.insert_one(company)
    return {"message": "Company added successfully"}