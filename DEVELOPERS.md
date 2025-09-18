# Developer Docs

## Project Overview
This project is an agentic investment research analyst platform. It uses MongoDB Atlas as a communal database and provides a FastAPI backend to extract data from MongoDB Atlas.

---

## MongoDB Atlas Setup
- The project uses a shared MongoDB Atlas cluster.
- The connection string is stored in the `.env` file as `ATLAS_URI`.
- For development, the cluster is open to all IPs (0.0.0.0/0), but for production, restrict IP access.

<!-- ON PRODUCTION, RESTRICT IP ACCESS 
- Each developer should:
  1. Request access to the Atlas cluster (ask the admin to create a user for you).
  2. Add your credentials to your local `.env` file (never commit real secrets).
  3. Example `.env` entry:
     ```
     ATLAS_URI=mongodb+srv://<username>:<password>@agenticaidb.d7py4bt.mongodb.net/test?retryWrites=true&w=majority&appName=agenticAIdb
     ``` -->

---

## FastAPI Backend
- The FastAPI app is located at `backend/app/fastapi_app.py`.
- To run the API locally:
  1. Install dependencies:
     ```sh
     pip install fastapi uvicorn pymongo python-dotenv
     ```
  2. Start the server:
     ```sh
     uvicorn backend.app.fastapi_app:app --reload
     ```
  3. Access the API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example Endpoints
- `GET /companies` — List all companies (returns ticker as a field)
- `GET /company/{ticker}` — Get a company by ticker symbol

---

## Environment Variables
- `.env` should never be committed to git. Use `.env.example` as a template.
- Required variables:
  - `FINNHUB_API_KEY` — Your Finnhub API key
  - `ATLAS_URI` — Your MongoDB Atlas connection string

---

## Branching Strategy
- Use feature branches for new features (e.g., `feat/add-fastapi-mongo-endpoints`).
- If your feature depends on an unmerged branch, branch off from it.
- Open pull requests for review before merging to `main`.

---

## Security Notes
- Never share admin credentials or commit secrets.
- Use individual database users for each developer.
- Restrict IP access in Atlas for production.

---

## Contact
For access or questions, contact the project admin.
