"""Bài Tập Cộng Điểm — Đo và tối ưu latency của hệ thống multi-agent (Stage 5).

So sánh A/B trên cùng một orchestrator (Law Agent graph), cùng model, cùng câu hỏi:

  [BEFORE] Topology gốc (tuần tự):
      analyze_law -> check_routing -> [call_tax || call_compliance] -> aggregate
      Critical path = analyze_law + check_routing + max(tax, compliance) + aggregate

  [AFTER]  Topology tối ưu (đưa analyze_law ra khỏi critical path):
      check_routing -> [analyze_law || call_tax || call_compliance] -> aggregate
      Critical path = check_routing + max(analyze_law, tax, compliance) + aggregate

`analyze_law` KHÔNG phụ thuộc vào kết quả routing hay specialist, nên có thể chạy
song song cùng tax/compliance thay vì chặn (block) chúng. Nhờ đó tiết kiệm trọn
một lượt gọi LLM (analyze_law) trên đường tới hạn.

Yêu cầu: registry + tax_agent + compliance_agent đang chạy (start_all.sh / start_all.ps1),
vì call_tax / call_compliance vẫn gọi thật qua A2A protocol.

Chạy:
    uv run python benchmark_latency.py
"""

import asyncio
import time
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from law_agent.graph import (
    LawState,
    aggregate,
    analyze_law,
    call_compliance,
    call_tax,
    check_routing,
    create_graph,  # graph gốc (BEFORE)
)

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


def route_optimized(state: LawState) -> list[Send]:
    """Dispatch analyze_law SONG SONG với specialist agents.

    Khác với route_to_subagents gốc: analyze_law được thêm vào danh sách Send
    nên nó chạy đồng thời với call_tax / call_compliance thay vì chạy trước đó.
    """
    sends: list[Send] = [Send("analyze_law", state)]
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    return sends


def create_optimized_graph():
    """Topology tối ưu: check_routing trước, rồi fan-out analyze_law + specialists."""
    graph = StateGraph(LawState)

    graph.add_node("check_routing", check_routing)
    graph.add_node("analyze_law", analyze_law)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_optimized,
        ["analyze_law", "call_tax", "call_compliance"],
    )
    graph.add_edge("analyze_law", "aggregate")
    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


def fresh_state() -> dict:
    return {
        "question": QUESTION,
        "context_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "delegation_depth": 0,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "tax_result": "",
        "compliance_result": "",
        "final_answer": "",
    }


async def timed_run(label: str, graph) -> float:
    print(f"\n[{label}] đang chạy...")
    t0 = time.perf_counter()
    result = await graph.ainvoke(fresh_state())
    dt = time.perf_counter() - t0
    ok = bool(result.get("final_answer"))
    print(f"[{label}] xong trong {dt:.2f}s — final_answer: {len(result.get('final_answer',''))} chars (ok={ok})")
    return dt


async def main():
    load_dotenv()

    print("=" * 70)
    print("ĐO LATENCY HỆ THỐNG MULTI-AGENT (Law Agent orchestrator)")
    print("=" * 70)
    print(f"Câu hỏi: {QUESTION}")

    before = await timed_run("BEFORE — tuần tự", create_graph())
    after = await timed_run("AFTER  — analyze_law song song", create_optimized_graph())

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"  BEFORE (gốc, tuần tự)        : {before:.2f}s")
    print(f"  AFTER  (analyze_law song song): {after:.2f}s")
    if before > 0:
        saved = before - after
        pct = saved / before * 100
        print(f"  Giảm được                    : {saved:.2f}s ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
