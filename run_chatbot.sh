#!/bin/bash

# LangGraph Equity Research Chatbot Launcher
# This script ensures the Python path is correctly set before launching Streamlit

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include backend/app
export PYTHONPATH="${SCRIPT_DIR}/backend/app:${PYTHONPATH}"

# Change to project root
cd "${SCRIPT_DIR}"

# Print environment info
echo "========================================="
echo "LangGraph Chatbot Launcher"
echo "========================================="
echo "Project root: ${SCRIPT_DIR}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import langgraph; print('✅ langgraph installed')" 2>/dev/null || {
    echo "❌ langgraph not found. Installing..."
    pip install langgraph==0.6.7
}

python -c "import streamlit; print('✅ streamlit installed')" 2>/dev/null || {
    echo "❌ streamlit not found. Installing..."
    pip install streamlit
}

echo ""
echo "Launching Streamlit chatbot..."
echo ""

# Run Streamlit
streamlit run frontend/equity_research_chatbot.py
