from typing import Any, Dict

from agents.utils.tools import web_search
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()


def _search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "").strip()
    if not query:
        return {"query": query, "results": []}
    result = web_search.invoke({"query": query})
    return {"query": query, "results": result.get("results", [])}


def build_websearch_graph():
    graph = StateGraph(dict)
    graph.add_node("search", _search_node)
    graph.set_entry_point("search")
    graph.add_edge("search", END)
    return graph.compile()


def run_websearch(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Helper to run the graph in one call."""
    app = build_websearch_graph()
    state = {"query": query, "max_results": max_results}
    final_state = app.invoke(state)
    return {"query": final_state.get("query", query), "results": final_state.get("results", [])}


if __name__ == "__main__":
    out = run_websearch("quantum stocks outlook", max_results=3)
    print({"urls": [r.get("url") for r in out.get("results", [])]})




