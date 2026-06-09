"""
Lab Assignment Day 09
Cải tiến Agent Day 08 (DrugLaw RAG) với Supervisor-Workers Pattern

Day 08 (cũ): 1 agent, pipeline tuần tự
    User → retrieve() → reorder → format_context → LLM → Answer

Day 09 (mới): Supervisor + 3 Workers song song
    User → Supervisor (route by intent)
              ↓ Send API (parallel dispatch)
         ┌────┴────────────────────┬────────────────────┐
    LegalDocWorker           NewsWorker           AnalysisWorker
    (văn bản pháp luật)    (tin tức thực tế)    (phân tích sâu)
         └─────────────────────────┴────────────────────┘
                              ↓
                         Aggregator
                              ↓
                         Final Answer

Cải tiến chính:
  - Chuyên môn hóa: mỗi worker có knowledge base + system prompt riêng
  - Song song: 3 workers chạy cùng lúc (LangGraph Send API)
  - Intent-aware: Supervisor chỉ kích hoạt workers cần thiết
  - Tổng hợp: Aggregator kết hợp 3 góc nhìn thành 1 câu trả lời hoàn chỉnh
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.constants import Send

# Dùng chung LLM config từ Day 09 (OpenRouter via common/llm.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.llm import get_llm


# ─── Knowledge Base (rút gọn từ Day 08 RAG corpus) ────────────────────────────

LEGAL_KB = [
    {
        "source": "Bộ luật Hình sự 2015 (sửa đổi 2017), Điều 249",
        "content": (
            "Tội tàng trữ trái phép chất ma tuý. Hình phạt: tù 1–5 năm (cơ bản). "
            "Số lượng lớn: 5–10 năm. Số lượng rất lớn: 10–15 năm. "
            "Đặc biệt lớn (>600g heroin hoặc cocaine): 15–20 năm hoặc tù chung thân."
        ),
    },
    {
        "source": "Bộ luật Hình sự 2015 (sửa đổi 2017), Điều 251",
        "content": (
            "Tội mua bán trái phép chất ma tuý. Cơ bản: tù 2–7 năm. "
            "Có tổ chức, lợi dụng chức vụ, tái phạm nguy hiểm: 7–15 năm. "
            "Đặc biệt nghiêm trọng: 15–20 năm, chung thân hoặc tử hình."
        ),
    },
    {
        "source": "Bộ luật Hình sự 2015 (sửa đổi 2017), Điều 246",
        "content": (
            "Tội sử dụng trái phép chất ma tuý. Lần đầu: xử lý hành chính, "
            "đưa đi cai nghiện bắt buộc 6–24 tháng (Luật PCMT 2021 Điều 96). "
            "Tái phạm lần 2: phạt tù 3 tháng đến 2 năm."
        ),
    },
    {
        "source": "Luật Phòng, chống ma tuý 2021",
        "content": (
            "Chất ma tuý cấm: heroin, cocaine, methamphetamine (đá), MDMA (ecstasy), "
            "cần sa, thuốc phiện, ketamine. Người nghiện có thể cai nghiện tự nguyện "
            "hoặc bị áp dụng cai nghiện bắt buộc tại cơ sở cấp tỉnh (12–24 tháng)."
        ),
    },
    {
        "source": "Nghị định 105/2021/NĐ-CP",
        "content": (
            "Xử phạt vi phạm hành chính về ma tuý. "
            "Tàng trữ dưới ngưỡng hình sự (<0.5g heroin): phạt 2–5 triệu + tịch thu. "
            "Tổ chức sử dụng ma tuý: 20–30 triệu đồng. "
            "Tàng trữ dụng cụ sử dụng ma tuý: 500K–1 triệu đồng."
        ),
    },
]

NEWS_KB = [
    {
        "source": "VnExpress, 2023",
        "content": (
            "Ca sĩ bị bắt tại TP.HCM với tang vật 2g methamphetamine (đá). "
            "Khởi tố tội tàng trữ theo Điều 249 BLHS. "
            "Bị tạm hoãn show diễn và đình chỉ hợp đồng quảng cáo ngay sau đó."
        ),
    },
    {
        "source": "Tuổi Trẻ, 2022",
        "content": (
            "Diễn viên và nhóm bạn bị bắt trong tiệc ma tuý tại khách sạn Hà Nội. "
            "5 người dương tính với MDMA và cocaine. "
            "Xử lý hành chính, đưa đi cai nghiện tự nguyện, sự nghiệp bị ảnh hưởng nặng."
        ),
    },
    {
        "source": "Thanh Niên, 2024",
        "content": (
            "Rapper bị cấm biểu diễn 24 tháng sau khi bắt quả tang với 5g cần sa. "
            "Xử phạt hành chính 3 triệu đồng. "
            "Cục Nghệ thuật Biểu diễn thu hồi giấy phép biểu diễn toàn quốc."
        ),
    },
    {
        "source": "Dân Trí, 2023",
        "content": (
            "Thống kê: hơn 20 nghệ sĩ Việt liên quan ma tuý bị xử lý trong 5 năm gần đây. "
            "60% xử lý hành chính (cai nghiện), 40% bị khởi tố hình sự. "
            "Hầu hết sự nghiệp kết thúc hoặc gián đoạn dài hạn."
        ),
    },
]


def _search(query: str, kb: list[dict], top_k: int = 3) -> list[dict]:
    """Keyword scoring search — thay thế ChromaDB vector search của Day 08."""
    q_words = set(query.lower().split())
    scored = []
    for item in kb:
        score = len(q_words & set(item["content"].lower().split()))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


# ─── State ────────────────────────────────────────────────────────────────────

def _last_wins(a: Any, b: Any) -> Any:
    return b if b is not None else a


class State(TypedDict):
    question: str
    active_workers: list[str]
    legal_answer: Annotated[str, _last_wins]
    news_answer: Annotated[str, _last_wins]
    analysis_answer: Annotated[str, _last_wins]
    final_answer: str


# ─── Supervisor ───────────────────────────────────────────────────────────────

def supervisor(state: State) -> dict:
    """
    Phân tích intent → chọn workers cần thiết.
    Không gọi LLM — rule-based để tiết kiệm latency (cải tiến so với Day 09 exercises).
    """
    q = state["question"].lower()

    legal_kws = [
        "hình phạt", "điều", "tội", "bộ luật", "phạt tù", "tàng trữ",
        "mua bán", "pháp luật", "nghị định", "quy định", "hành chính",
        "hình sự", "cai nghiện", "ma tu",
    ]
    news_kws = [
        "nghệ sĩ", "ca sĩ", "diễn viên", "rapper", "sao", "nổi tiếng",
        "bị bắt", "vụ án", "tin tức", "thống kê",
    ]
    analysis_kws = [
        "phân tích", "giải thích", "so sánh", "tại sao",
        "hậu quả", "ảnh hưởng", "khuyến nghị",
    ]

    workers = []
    if any(kw in q for kw in legal_kws):
        workers.append("legal_doc_worker")
    if any(kw in q for kw in news_kws):
        workers.append("news_worker")
    if any(kw in q for kw in analysis_kws):
        workers.append("analysis_worker")

    # Câu hỏi chung hoặc không match → kích hoạt tất cả
    if not workers:
        workers = ["legal_doc_worker", "news_worker", "analysis_worker"]

    print(f"  [Supervisor] Workers: {workers}")
    return {"active_workers": workers}


def supervisor_route(state: State) -> list[Send]:
    """Send API dispatch — tất cả workers trong active_workers chạy song song."""
    return [Send(worker, state) for worker in state["active_workers"]]


# ─── Worker 1: Legal Document Worker ─────────────────────────────────────────

def legal_doc_worker(state: State) -> dict:
    """
    Tra cứu văn bản pháp luật ma tuý.
    Day 08: retrieve() + reorder_for_llm() + format_context() gộp trong 1 pipeline.
    Day 09: tách riêng, chạy song song với workers khác.
    """
    chunks = _search(state["question"], LEGAL_KB, top_k=3)
    context = "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in chunks)

    if not context.strip():
        return {"legal_answer": "Không tìm thấy văn bản pháp luật liên quan."}

    llm = get_llm()
    prompt = f"""Bạn là chuyên gia pháp luật phòng chống ma tuý Việt Nam.
Dựa vào văn bản pháp luật dưới đây, trả lời ngắn gọn (tối đa 120 từ).
Mỗi thông tin PHẢI có citation [Nguồn, Điều X].

Context:
{context}

Câu hỏi: {state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"legal_answer": response.content}


# ─── Worker 2: News Context Worker ───────────────────────────────────────────

def news_worker(state: State) -> dict:
    """
    Tra cứu tin tức và các vụ án nghệ sĩ thực tế.
    Chỉ chạy khi câu hỏi có từ khóa liên quan nghệ sĩ/tin tức.
    """
    chunks = _search(state["question"], NEWS_KB, top_k=2)

    if not chunks:
        return {"news_answer": "Không tìm thấy tin tức liên quan trong knowledge base."}

    context = "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in chunks)

    llm = get_llm()
    prompt = f"""Bạn là nhà báo pháp lý chuyên về các vụ án nghệ sĩ liên quan ma tuý.
Cung cấp ngữ cảnh thực tế ngắn gọn (tối đa 100 từ). Giữ nguyên citation [Nguồn].

Context:
{context}

Câu hỏi: {state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"news_answer": response.content}


# ─── Worker 3: Deep Analysis Worker ──────────────────────────────────────────

def analysis_worker(state: State) -> dict:
    """
    Phân tích sâu: hậu quả pháp lý, so sánh, khuyến nghị.
    Tính năng hoàn toàn mới, không có trong Day 08.
    """
    llm = get_llm()
    prompt = f"""Bạn là luật sư phân tích chính sách phòng chống ma tuý Việt Nam.
Phân tích các khía cạnh sau (tối đa 120 từ):
- Hậu quả pháp lý (hình sự vs hành chính, tùy tình huống)
- So sánh với trường hợp tương tự
- Khuyến nghị thực tế

Câu hỏi: {state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"analysis_answer": response.content}


# ─── Aggregator ───────────────────────────────────────────────────────────────

def aggregator(state: State) -> dict:
    """
    Tổng hợp kết quả từ tất cả workers thành câu trả lời cuối.
    Tương tự aggregate_results trong exercise_4_multiagent.py.
    """
    sections = []
    if state.get("legal_answer"):
        sections.append(f"## Quy định pháp luật\n{state['legal_answer']}")
    if state.get("news_answer"):
        sections.append(f"## Ngữ cảnh thực tế\n{state['news_answer']}")
    if state.get("analysis_answer"):
        sections.append(f"## Phân tích chuyên sâu\n{state['analysis_answer']}")

    # Chỉ 1 worker → trả thẳng không cần tổng hợp LLM
    if len(sections) == 1:
        return {"final_answer": sections[0].split("\n", 1)[1].strip()}

    combined = "\n\n---\n\n".join(sections)

    llm = get_llm()
    prompt = f"""Tổng hợp các phân tích sau thành 1 câu trả lời mạch lạc (tối đa 250 từ, tiếng Việt).
Giữ nguyên tất cả citation.

{combined}

Câu hỏi gốc: {state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)

    graph.add_node("supervisor", supervisor)
    graph.add_node("legal_doc_worker", legal_doc_worker)
    graph.add_node("news_worker", news_worker)
    graph.add_node("analysis_worker", analysis_worker)
    graph.add_node("aggregator", aggregator)

    graph.set_entry_point("supervisor")

    # Supervisor dùng Send API → dispatch song song đến các workers
    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        ["legal_doc_worker", "news_worker", "analysis_worker"],
    )

    # Tất cả workers fan-in vào aggregator
    graph.add_edge("legal_doc_worker", "aggregator")
    graph.add_edge("news_worker", "aggregator")
    graph.add_edge("analysis_worker", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    questions = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý gần đây?",
        "Phân tích hậu quả pháp lý khi nghệ sĩ bị bắt vì ma tuý - so sánh hình sự và hành chính",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"CAU HOI: {q}")
        print("=" * 70)

        t0 = time.time()
        result = app.invoke({"question": q})
        elapsed = time.time() - t0

        print(f"\n[Workers: {', '.join(result.get('active_workers', []))} | {elapsed:.1f}s]")
        print(f"\nKET QUA:\n{result['final_answer']}")
