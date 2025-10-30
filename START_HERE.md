# 🚀 Quick Start - Equity Research Chatbot

## Run the Chatbot (3 Steps)

### Step 1: Set Python Path
```bash
export PYTHONPATH="${PWD}/backend/app:${PYTHONPATH}"
```

### Step 2: Ensure Services are Running

**Option A - All services with Docker:**
```bash
docker-compose up -d
```

**Option B - Manual (Recommended for testing):**

Terminal 1 - Ollama:
```bash
ollama serve
```

Terminal 2 - FastAPI:
```bash
python -m uvicorn backend.app.api_controller:app --reload
```

### Step 3: Launch Chatbot
```bash
streamlit run frontend/equity_research_chatbot.py
```

The chatbot will open in your browser at http://localhost:8501

---

## Alternative: Use Launcher Script

```bash
# Make executable (first time only)
chmod +x run_chatbot.sh

# Run
./run_chatbot.sh
```

---

## Example Queries to Try

### Company Analysis (Full Pipeline)
- "Analyze Tesla stock"
- "Give me an investment report for Apple"
- "Should I invest in Microsoft?"

### Market Analysis
- "What are current inflation trends?"
- "How is the tech sector performing?"
- "Analyze S&P 500 trends"

### Educational Content
- "What is P/E ratio?"
- "Explain earnings per share"
- "How does market cap work?"

---

## Troubleshooting

### Error: "Module not found"
```bash
# Set PYTHONPATH again
export PYTHONPATH="${PWD}/backend/app:${PYTHONPATH}"
```

### Error: "Cannot connect to Ollama"
```bash
# Start Ollama
ollama serve

# Pull model (in another terminal)
ollama pull llama3.1
```

### Error: "OpenAI API key not found"
```bash
# Check .env file exists and has your API key
cat .env | grep OPENAI_API_KEY
```

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## What to Expect

1. **First launch**: May take 30-60 seconds to initialize
2. **First query**: May take 20-30 seconds (model loading)
3. **Subsequent queries**: Much faster (5-15 seconds)

**Processing times by intent:**
- Irrelevant: ~2 seconds
- Education: ~5 seconds
- Market: ~10 seconds
- Company (full analysis): ~25 seconds

---

## Success Indicators

✅ Chatbot interface loads at http://localhost:8501
✅ Example queries return structured responses
✅ Different intents route correctly
✅ PDF reports generate in `research_outputs/`
✅ No errors in terminal logs

---

## Full Documentation

- **Quick Start Guide**: [QUICKSTART_LANGGRAPH.md](QUICKSTART_LANGGRAPH.md)
- **Technical Documentation**: [LANGGRAPH_PIPELINE.md](LANGGRAPH_PIPELINE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Visual Diagrams**: [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md)

---

## Need Help?

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
2. Review error messages in terminal
3. Verify all prerequisites are installed
4. Ensure services are running (Ollama, FastAPI, MongoDB)

**Enjoy your AI-powered equity research assistant!** 🎉
