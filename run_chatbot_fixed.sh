#!/bin/bash

# Equity Research Chatbot Launcher (Fixed for Virtual Environment)

cd "$(dirname "$0")"

echo "========================================="
echo "Equity Research Chatbot Launcher"
echo "========================================="
echo ""

# Check if agenticenv exists
if [ -d "agenticenv" ]; then
    echo "✅ Found agenticenv virtual environment"
    echo "Activating agenticenv..."
    source agenticenv/bin/activate
else
    echo "❌ agenticenv not found"
    echo "Creating new virtual environment..."
    python3 -m venv agenticenv
    source agenticenv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Verify Python
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Install/check langgraph
echo "Checking langgraph installation..."
python -c "import langgraph; print('✅ langgraph installed')" 2>/dev/null || {
    echo "Installing langgraph..."
    pip install langgraph==0.6.7
}

# Set PYTHONPATH
export PYTHONPATH="${PWD}/backend/app:${PYTHONPATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

echo "Starting Streamlit chatbot..."
echo "========================================="
echo ""

# Run Streamlit
streamlit run frontend/equity_research_chatbot.py
