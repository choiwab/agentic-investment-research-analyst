"""
LangGraph Equity Research Chatbot Launcher

This script properly sets up the Python path and launches the Streamlit chatbot.
"""

import os
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add backend/app to Python path
backend_app_path = PROJECT_ROOT / "backend" / "app"
sys.path.insert(0, str(backend_app_path))

print("=" * 80)
print("LANGGRAPH EQUITY RESEARCH CHATBOT")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")
print(f"Backend path: {backend_app_path}")
print()

# Check dependencies
print("Checking dependencies...")
try:
    import langgraph
    print(f"✅ langgraph {langgraph.__version__}")
except ImportError:
    print("❌ langgraph not found")
    print("   Installing: pip install langgraph==0.6.7")
    os.system("pip install langgraph==0.6.7")

try:
    import streamlit
    print(f"✅ streamlit {streamlit.__version__}")
except ImportError:
    print("❌ streamlit not found")
    print("   Installing: pip install streamlit")
    os.system("pip install streamlit")

print()
print("Launching Streamlit chatbot...")
print("=" * 80)
print()

# Launch Streamlit
os.system(f"streamlit run {PROJECT_ROOT}/frontend/equity_research_chatbot.py")
