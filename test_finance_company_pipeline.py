#!/usr/bin/env python
"""
Comprehensive test for the finance-company pipeline.
Tests the full orchestration: preprocessing → news_scraper → metric_extractor → sentiment_extractor → research_compiler
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add paths
project_root = Path(__file__).parent
backend_app_path = project_root / "backend" / "app"
backend_agents_path = backend_app_path / "agents"

for path in [str(backend_app_path), str(backend_agents_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.equity_research_graph import EquityResearchOrchestrator

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_state_summary(state: Dict[str, Any]):
    """Print a summary of the state at each stage."""
    print(f"\n📍 Current Node: {state.get('current_node', 'N/A')}")
    print(f"🎯 Intent: {state.get('intent', 'N/A')}")
    print(f"🏢 Ticker: {state.get('ticker', 'N/A')}")

    # Preprocessing outputs
    if state.get('timeframe'):
        print(f"⏰ Timeframe: {state.get('timeframe')}")
    if state.get('metrics'):
        print(f"📊 Metrics: {', '.join(state.get('metrics', []))}")
    if state.get('url'):
        print(f"🔗 URL: {state.get('url')[:80]}...")

    # News scraper outputs
    if state.get('qualitative_summary'):
        summary = state.get('qualitative_summary', '')[:150]
        print(f"📰 Qualitative Summary: {summary}...")
    if state.get('insight_outlook'):
        outlook = state.get('insight_outlook', '')[:150]
        print(f"🔮 Insight Outlook: {outlook}...")

    # Metric extractor outputs
    if state.get('financials'):
        print(f"💰 Financials: {list(state.get('financials', {}).keys())}")
    if state.get('valuation'):
        print(f"📈 Valuation: {list(state.get('valuation', {}).keys())}")
    if state.get('metric_evaluation'):
        print(f"📊 Metric Evaluation: Available")

    # Sentiment extractor outputs
    if state.get('sentiment_label'):
        print(f"💭 Sentiment: {state.get('sentiment_label')} ({state.get('sentiment_confidence', 0):.2%} confidence)")

    # Research compiler outputs
    if state.get('executive_summary'):
        exec_summary = state.get('executive_summary', '')[:150]
        print(f"📝 Executive Summary: {exec_summary}...")
    if state.get('recommendation'):
        print(f"🎯 Recommendation: {state.get('recommendation')}")
    if state.get('price_target'):
        print(f"💵 Price Target: {state.get('price_target')}")

    # Errors and warnings
    if state.get('errors'):
        print(f"\n⚠️  ERRORS ({len(state.get('errors', []))}):")
        for error in state.get('errors', []):
            print(f"   - {error}")
    if state.get('warnings'):
        print(f"\n⚠️  WARNINGS ({len(state.get('warnings', []))}):")
        for warning in state.get('warnings', []):
            print(f"   - {warning}")

def test_finance_company_pipeline():
    """Test the complete finance-company pipeline with Tesla stock."""

    print_section("FINANCE-COMPANY PIPELINE TEST")
    print("\nQuery: 'Give me a comprehensive analysis on Tesla stock'")
    print("Expected flow: preprocessing → news_scraper → metric_extractor → sentiment_extractor → research_compiler")

    try:
        # Initialize orchestrator
        print("\n🏗️  Initializing orchestrator...")
        orchestrator = EquityResearchOrchestrator()
        print("✅ Orchestrator initialized")

        # Run the pipeline
        print_section("RUNNING PIPELINE")
        query = "Give me a comprehensive analysis on Tesla stock"

        result = orchestrator.run(query)

        # Analyze results
        print_section("PIPELINE RESULTS")
        print_state_summary(result)

        # Validate pipeline execution
        print_section("VALIDATION")

        checks = []

        # Check 1: Intent classification
        intent_correct = result.get('intent') == 'finance-company'
        checks.append(("Intent classified as finance-company", intent_correct))

        # Check 2: Ticker extraction
        ticker_correct = result.get('ticker') == 'TSLA'
        checks.append(("Ticker extracted as TSLA", ticker_correct))

        # Check 3: News scraper ran
        news_ran = bool(result.get('qualitative_summary') or result.get('insight_outlook'))
        checks.append(("News scraper executed", news_ran))

        # Check 4: Metric extractor ran
        metrics_ran = bool(result.get('financials') or result.get('valuation'))
        checks.append(("Metric extractor executed", metrics_ran))

        # Check 5: Sentiment extractor ran
        sentiment_ran = bool(result.get('sentiment_label'))
        checks.append(("Sentiment extractor executed", sentiment_ran))

        # Check 6: Research compiler ran
        compiler_ran = bool(result.get('executive_summary') or result.get('recommendation'))
        checks.append(("Research compiler executed", compiler_ran))

        # Check 7: Error handling
        no_critical_errors = len(result.get('errors', [])) == 0
        checks.append(("No critical errors", no_critical_errors))

        # Print validation results
        print()
        passed = 0
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"{status} {check_name}")
            if check_result:
                passed += 1

        print(f"\n📊 Pipeline Validation: {passed}/{len(checks)} checks passed")

        # Error analysis
        if result.get('errors'):
            print_section("ERROR ANALYSIS")
            for i, error in enumerate(result.get('errors', []), 1):
                print(f"\n{i}. {error}")

                # Categorize errors
                if "preprocessing" in error.lower():
                    print("   Category: Preprocessing")
                    print("   Impact: Pipeline may not route correctly")
                elif "news" in error.lower():
                    print("   Category: News Scraping")
                    print("   Impact: Missing news context for analysis")
                elif "metric" in error.lower():
                    print("   Category: Metric Extraction")
                    print("   Impact: Missing financial data")
                elif "sentiment" in error.lower():
                    print("   Category: Sentiment Analysis")
                    print("   Impact: Missing sentiment insights")
                elif "compiler" in error.lower():
                    print("   Category: Research Compilation")
                    print("   Impact: Final report may be incomplete")

        # Data flow analysis
        print_section("DATA FLOW ANALYSIS")

        print("\n1. Preprocessing → News Scraper:")
        if result.get('url'):
            print(f"   ✅ URL passed: {result.get('url')[:60]}...")
        else:
            print("   ⚠️  No URL passed (news scraper may fail)")

        print("\n2. News Scraper → Metric Extractor:")
        if result.get('qualitative_summary'):
            print(f"   ✅ News context available ({len(result.get('qualitative_summary', ''))} chars)")
        else:
            print("   ⚠️  No news context (metric extractor continues without it)")

        print("\n3. News + Metrics → Sentiment Extractor:")
        text_available = bool(result.get('qualitative_summary') or result.get('insight_outlook'))
        if text_available:
            print("   ✅ Text available for sentiment analysis")
        else:
            print("   ⚠️  No text for sentiment analysis")

        print("\n4. All Data → Research Compiler:")
        data_sources = []
        if result.get('financials'): data_sources.append("financials")
        if result.get('valuation'): data_sources.append("valuation")
        if result.get('qualitative_summary'): data_sources.append("news")
        if result.get('sentiment_label'): data_sources.append("sentiment")

        if data_sources:
            print(f"   ✅ Data sources available: {', '.join(data_sources)}")
        else:
            print("   ⚠️  Limited data for compilation")

        # Overall assessment
        print_section("OVERALL ASSESSMENT")

        if passed == len(checks) and no_critical_errors:
            print("\n✅ Pipeline executed successfully!")
            print("   All nodes ran correctly and data was passed through properly.")
        elif passed >= len(checks) * 0.7:
            print("\n⚠️  Pipeline completed with some issues")
            print("   Most nodes executed but there were errors or missing data.")
            print("   The pipeline handled errors gracefully and continued execution.")
        else:
            print("\n❌ Pipeline execution had significant issues")
            print("   Multiple nodes failed or critical data was missing.")
            print("   Review errors above to diagnose the problem.")

        # Save full results to file
        output_file = project_root / "test_results_finance_company.json"
        with open(output_file, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_result = {}
            for key, value in result.items():
                try:
                    json.dumps(value)
                    serializable_result[key] = value
                except:
                    serializable_result[key] = str(value)

            json.dump(serializable_result, f, indent=2)

        print(f"\n📄 Full results saved to: {output_file}")

        print("\n" + "=" * 80)

        return passed == len(checks)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling with invalid input."""

    print_section("ERROR HANDLING TEST")
    print("\nQuery: 'Analyze INVALIDTICKER123' (non-existent ticker)")
    print("Expected: Pipeline should handle gracefully and continue")

    try:
        orchestrator = EquityResearchOrchestrator()
        result = orchestrator.run("Analyze INVALIDTICKER123 stock")

        print("\n✅ Pipeline completed without crashing")
        print(f"   Errors logged: {len(result.get('errors', []))}")
        print(f"   Intent: {result.get('intent')}")

        if result.get('errors'):
            print("\n   Error messages:")
            for error in result.get('errors', []):
                print(f"   - {error[:100]}")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline crashed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "EQUITY RESEARCH PIPELINE TEST SUITE")
    print("=" * 80)

    # Test 1: Full finance-company pipeline
    test1_passed = test_finance_company_pipeline()

    # Test 2: Error handling
    test2_passed = test_error_handling()

    # Final summary
    print_section("TEST SUITE SUMMARY")
    print(f"\n1. Finance-Company Pipeline: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"2. Error Handling: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    all_passed = test1_passed and test2_passed

    if all_passed:
        print("\n✅ All tests passed! The pipeline is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")

    print("\n" + "=" * 80)

    sys.exit(0 if all_passed else 1)
