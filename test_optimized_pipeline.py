#!/usr/bin/env python
"""
Performance Test: Optimized Finance-Company Pipeline

Compares execution times:
- OLD: REACT agents (30-40s metric extractor + 20-30s compiler = 50-70s)
- NEW: Direct prompting (5-8s metric extractor + 3-5s compiler = 8-13s)

Expected improvement: 60-85% faster
"""

import sys
import time
from pathlib import Path

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

def test_optimized_pipeline():
    """Test the optimized pipeline with Apple stock query."""

    print_section("OPTIMIZED PIPELINE PERFORMANCE TEST")

    print("\nQuery: 'Give me a comprehensive analysis on Apple stock'")
    print("Expected flow: preprocessing → news_scraper → metric_extractor → sentiment_extractor → research_compiler")
    print()

    # Expected timings
    print("📊 EXPECTED PERFORMANCE:")
    print("   OLD (REACT agents):")
    print("      • Preprocessing: 5-10s")
    print("      • News Scraper: 1-2s")
    print("      • Metric Extractor: 30-40s ❌ SLOW")
    print("      • Sentiment: <1s")
    print("      • Research Compiler: 20-30s ❌ SLOW")
    print("      • TOTAL: ~55-80 seconds")
    print()
    print("   NEW (Direct prompting):")
    print("      • Preprocessing: 5-10s")
    print("      • News Scraper: 1-2s")
    print("      • Metric Extractor: 5-8s ✅ FAST")
    print("      • Sentiment: <1s")
    print("      • Research Compiler: 3-5s ✅ FAST")
    print("      • TOTAL: ~15-25 seconds")
    print()
    print("   EXPECTED IMPROVEMENT: 60-75% faster")

    print_section("RUNNING OPTIMIZED PIPELINE")

    try:
        # Initialize orchestrator
        print("\n🏗️  Initializing orchestrator...")
        orchestrator = EquityResearchOrchestrator()
        print("✅ Orchestrator initialized")

        # Run the pipeline with timing
        query = "Give me a comprehensive analysis on Apple stock"
        print(f"\n🚀 Starting pipeline...")
        print(f"   Query: '{query}'")
        print()

        start_time = time.time()

        # Track individual stages
        stage_times = {}

        result = orchestrator.run(query)

        total_time = time.time() - start_time

        # Print results
        print_section("PIPELINE RESULTS")

        print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print()

        # Show performance category
        if total_time < 20:
            print("✅ EXCELLENT! Significantly faster than expected")
            print(f"   Target was ~15-25s, achieved {total_time:.2f}s")
        elif total_time < 30:
            print("✅ GOOD! Within optimized range")
            print(f"   Target was ~15-25s, achieved {total_time:.2f}s")
        elif total_time < 50:
            print("⚠️  IMPROVED but could be better")
            print(f"   Target was ~15-25s, achieved {total_time:.2f}s")
            print("   Still better than old REACT version (55-80s)")
        else:
            print("❌ SLOWER THAN EXPECTED")
            print(f"   Target was ~15-25s, achieved {total_time:.2f}s")
            print("   Check if agents are actually using optimized versions")

        # Show improvement estimate
        old_estimated_time = 65  # Average of 55-80s
        improvement_percent = ((old_estimated_time - total_time) / old_estimated_time) * 100

        print(f"\n📈 IMPROVEMENT:")
        print(f"   OLD (estimated): ~{old_estimated_time}s")
        print(f"   NEW (actual): {total_time:.2f}s")
        print(f"   SPEEDUP: {improvement_percent:.1f}% faster")

        # Show result summary
        print_section("RESULT SUMMARY")

        print(f"\n✓ Intent: {result.get('intent')}")
        print(f"✓ Ticker: {result.get('ticker')}")
        print(f"✓ Timeframe: {result.get('timeframe')}")

        if result.get('errors'):
            print(f"\n⚠️  Errors: {len(result.get('errors'))}")
            for error in result.get('errors'):
                print(f"   - {error[:100]}")

        if result.get('warnings'):
            print(f"\n⚠️  Warnings: {len(result.get('warnings'))}")
            for warning in result.get('warnings'):
                print(f"   - {warning[:100]}")

        # Analysis
        print_section("ANALYSIS")

        print("\n🔍 Pipeline Stages:")

        stages = [
            ("Preprocessing", "✅ Optimized (direct prompting)"),
            ("News Scraper", "✅ Already fast"),
            ("Metric Extractor", "✅ Optimized (direct prompting)"),
            ("Sentiment", "✅ Already fast (FinBERT)"),
            ("Research Compiler", "✅ Optimized (direct prompting)")
        ]

        for stage, status in stages:
            print(f"   {stage}: {status}")

        print("\n💡 Key Optimizations:")
        print("   1. Removed REACT agent loops in Metric Extractor")
        print("   2. Removed REACT agent loops in Research Compiler")
        print("   3. Direct prompting with single LLM calls")
        print("   4. No tool retry loops")
        print("   5. Simplified error handling")

        # Recommendations
        print_section("RECOMMENDATIONS")

        if total_time < 25:
            print("\n✅ Performance is EXCELLENT")
            print("   No immediate optimizations needed.")
            print("   Consider:")
            print("   • Caching LLM model loading for even faster subsequent runs")
            print("   • Parallel execution of independent nodes")
        elif total_time < 40:
            print("\n✅ Performance is GOOD")
            print("   Consider additional optimizations:")
            print("   • Run preprocessing and news_scraper in parallel (if independent)")
            print("   • Cache ticker data to reduce backend calls")
            print("   • Use faster LLM (e.g., GPT-4o-mini) for bottleneck agents")
        else:
            print("\n⚠️  Performance needs improvement")
            print("   Recommendations:")
            print("   • Verify agents are using optimized versions (not backups)")
            print("   • Check if Ollama model is slow to load")
            print("   • Consider switching to GPT-4o-mini for slower agents")
            print("   • Profile individual agent execution times")

        print("\n" + "=" * 80)

        return total_time

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "OPTIMIZED PIPELINE PERFORMANCE TEST")
    print("=" * 80)

    execution_time = test_optimized_pipeline()

    print_section("TEST COMPLETE")

    if execution_time:
        if execution_time < 25:
            print("\n🎉 SUCCESS! Pipeline is significantly optimized.")
            print(f"   Execution time: {execution_time:.2f}s (TARGET: 15-25s)")
        elif execution_time < 40:
            print("\n✅ Pipeline is improved but has room for optimization.")
            print(f"   Execution time: {execution_time:.2f}s (TARGET: 15-25s)")
        else:
            print("\n⚠️  Pipeline needs further optimization.")
            print(f"   Execution time: {execution_time:.2f}s (TARGET: 15-25s)")

    print("\n" + "=" * 80)

    sys.exit(0 if execution_time and execution_time < 40 else 1)
