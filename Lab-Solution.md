# Lab Solution — Day 09: Multi-Agent MCP & A2A

**Học viên:** vanhung71388@gmail.com  
**Ngày:** 2026-06-09  
**Model:** `openai/gpt-oss-120b:free` (OpenRouter free-tier)

---

## Phần 1: Direct LLM Calling

### Bài Tập 1.1 — Thay đổi câu hỏi

File `stages/stage_1_direct_llm/main.py`, sửa biến `QUESTION`:

```python
QUESTION = "Theo pháp luật Việt Nam, hành vi tàng trữ trái phép chất ma tuý bị xử lý như thế nào?"
```

Chạy: `uv run python stages/stage_1_direct_llm/main.py`

**Nhận xét:** LLM trả lời dựa hoàn toàn vào training data, không tra cứu được văn bản pháp luật thực tế.

### Bài Tập 1.2 — Thêm Temperature Control

File `common/llm.py`:

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.3")),  # BÀI TẬP 1.2
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "500")),
    )
```

**Giải thích:** `temperature=0.3` — thấp để output factual và ổn định, phù hợp tư vấn pháp lý. Đọc từ `.env` để dễ thay đổi mà không sửa code.

---

## Phần 2: LLM + RAG & Tools

### Bài Tập 2.1 — Thêm knowledge base entry (Luật lao động)

File `exercises/exercise_2_tools.py` — thêm vào `LEGAL_KNOWLEDGE`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

### Bài Tập 2.2 — Tạo tool `check_statute_of_limitations`

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")

tools = [search_legal_knowledge, check_statute_of_limitations]
llm_with_tools = llm.bind_tools(tools)
```

Thêm branch xử lý trong vòng lặp tool_calls:
```python
elif tool_call["name"] == "check_statute_of_limitations":
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

**Kết quả:** LLM tự nhận biết gọi `check_statute_of_limitations(case_type="contract")` mà không cần hướng dẫn — Function Calling tự động.

---

## Phần 3: Single Agent với ReAct

### Bài Tập 3.1 — Thêm tool tra cứu án lệ

File `stages/stage_3_single_agent/main.py`:

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.

    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

**Nhận xét:** ReAct agent tự động quyết định khi nào gọi `search_case_law` vs `search_legal_knowledge`, chuỗi tool calls không cần code thủ công.

### Bài Tập 3.2 — Debug agent reasoning

Thêm `verbose=True` trong `create_react_agent()` để xem chi tiết Thought → Action → Observation.

---

## Phần 4: Multi-Agent In-Process

### Bài Tập 4.1 — Implement `privacy_agent`

File `exercises/exercise_4_multiagent.py`:

```python
def privacy_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}
Tập trung: GDPR, data protection, privacy rights, data breach notification, hình phạt vi phạm."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

### Bài Tập 4.2 — Conditional routing

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    return tasks if tasks else [Send("aggregate_results", state)]
```

Thêm vào graph:
```python
graph.add_node("privacy_agent", privacy_agent)
graph.add_conditional_edges(
    "law_agent",
    check_routing,
    ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
)
graph.add_edge("privacy_agent", "aggregate_results")
```

**Bug đã fix:** Skeleton gốc đăng ký `check_routing` nhầm thành node (trả về `list[Send]` thay vì `dict`). Phải dùng làm path function trong `add_conditional_edges` — đây là pattern đúng của LangGraph Send API.

---

## Phần 5: Distributed A2A System

### Bài Tập 5.1 — Trace request flow

Trong logs, tìm `trace_id` (UUID được tạo lúc request vào Customer Agent, truyền qua tất cả hops):

```
Sequence diagram thực tế:
User → Customer Agent (10100)
         ↓ discover("legal_question") → Registry (10000)
         ↓ POST /tasks → Law Agent (10101)
                           ↓ analyze_law (LLM)
                           ↓ check_routing (LLM)
                           ↓ Send("tax_agent") + Send("compliance_agent") [song song]
                           ↓ aggregate (LLM)
         ← response
```

### Bài Tập 5.2 — Test dynamic discovery

Khi dừng Tax Agent → Registry không còn entry `tax-agent` → Law Agent gọi `/discover/tax_question` trả về lỗi 404 → Law Agent bỏ qua tax analysis, chỉ trả về compliance result.

**Kết luận:** Hệ thống degraded gracefully — không crash, chỉ thiếu phần tax.

### Bài Tập 5.3 — Modify agent behavior

Sửa `tax_agent/graph.py` — system prompt: thêm "Trả lời NGẮN GỌN, tối đa 3 câu." Restart tax agent: `python -m tax_agent`. Kết quả trả về ngắn hơn đáng kể.

---

## Bài Tập Cộng Điểm

### 1. Interactive HTML Demo

File `agent_demo.html` — standalone demo với 4 tabs:
- Stage 4: LangGraph StateGraph diagram + animated walkthrough
- Stage 5: A2A distributed simulation với live message log
- Latency Benchmark: animated bar chart (267s vs 119s)
- Code Patterns: reference snippets cho tất cả stages

### 2. Latency & Tối Ưu Hóa

| | Sequential | Parallel (asyncio.gather / Send API) |
|---|---|---|
| Latency thực đo | 267.21s | 119.20s |
| Giảm | — | **148s (55.4%)** |

**Phương án:** Dispatch Tax + Compliance song song thay vì tuần tự.  
**File demo:** `benchmark_parallel.py`  
**Phân tích đầy đủ:** `BONUS_LATENCY.md`

**Lý giải:** Tax và Compliance agent không phụ thuộc nhau — latency từ `t1+t2` giảm xuống `max(t1,t2)`.
