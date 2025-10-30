#!/usr/bin/env python
"""
Quick test script for Ollama-based preprocessing agent.
Tests intent classification with different query types.
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
backend_app_path = project_root / "backend" / "app"
backend_agents_path = backend_app_path / "agents"

for path in [str(backend_app_path), str(backend_agents_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.preprocessor import PreprocessAgent

def test_preprocessing():
    """Test the preprocessing agent with different query types."""

    print("=" * 80)
    print("OLLAMA PREPROCESSING AGENT TEST")
    print("=" * 80)
    print()

    agent = PreprocessAgent(model="llama3.1")

    test_queries = [
        {
            "query": "Give me a comprehensive analysis on Tesla stock",
            "expected_intent": "finance-company",
            "expected_ticker": "TSLA"
        },
        {
            "query": "Should I invest in Apple?",
            "expected_intent": "finance-company",
            "expected_ticker": "AAPL"
        },
        {
            "query": "What are current inflation trends?",
            "expected_intent": "finance-market",
            "expected_ticker": None
        },
        {
            "query": "What is P/E ratio?",
            "expected_intent": "finance-education",
            "expected_ticker": None
        },
        {
            "query": "Tell me a joke about dogs",
            "expected_intent": "irrelevant",
            "expected_ticker": None
        }
    ]

    results = []

    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}")
        print(f"Query: \"{test['query']}\"")

        try:
            result = agent.run({"query": test["query"]})

            intent_match = result["intent"] == test["expected_intent"]
            ticker_match = result["ticker"] == test["expected_ticker"]

            status = "✅ PASS" if intent_match else "❌ FAIL"

            print(f"Expected Intent: {test['expected_intent']}")
            print(f"Actual Intent:   {result['intent']} {status}")

            if test["expected_ticker"]:
                ticker_status = "✅" if ticker_match else "❌"
                print(f"Expected Ticker: {test['expected_ticker']}")
                print(f"Actual Ticker:   {result['ticker']} {ticker_status}")

            results.append({
                "query": test["query"],
                "passed": intent_match and (ticker_match if test["expected_ticker"] else True)
            })

        except Exception as e:
            print(f"❌ ERROR: {str(e)[:100]}")
            results.append({"query": test["query"], "passed": False})

        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print()

    if passed == total:
        print("✅ All tests passed! The preprocessing agent is working correctly with Ollama.")
    else:
        print("⚠️  Some tests failed. Review the results above.")
        for r in results:
            if not r["passed"]:
                print(f"  - Failed: {r['query']}")

    print()
    print("=" * 80)

    return passed == total

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)
