"""
Simple test script for the LangGraph pipeline
Tests basic functionality without Streamlit
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend" / "app"
sys.path.insert(0, str(backend_path))

print("=" * 80)
print("TESTING LANGGRAPH PIPELINE")
print("=" * 80)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from agents.equity_research_graph import EquityResearchOrchestrator
    print("✅ Successfully imported EquityResearchOrchestrator")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize orchestrator
print("\n2. Initializing orchestrator...")
try:
    orchestrator = EquityResearchOrchestrator()
    print("✅ Orchestrator initialized successfully")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with a simple irrelevant query (should be fastest)
print("\n3. Testing with irrelevant intent...")
try:
    result = orchestrator.run("Tell me a joke")
    print(f"✅ Query processed successfully")
    print(f"   Intent: {result.get('intent')}")
    print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
    print(f"   Current node: {result.get('current_node')}")
except Exception as e:
    print(f"❌ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ BASIC PIPELINE TEST PASSED!")
print("=" * 80)
print("\nThe pipeline is working correctly. The Streamlit app should work now.")
print("\nTo run the chatbot:")
print("  streamlit run frontend/equity_research_chatbot.py")
