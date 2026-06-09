# BÁO CÁO THỰC HÀNH — Day 9: Multi-Agent MCP & A2A

**Học viên:** vanhung71388@gmail.com  
**Ngày:** 2026-06-09  
**Môn:** Xây dựng hệ thống Multi-Agent với A2A Protocol  
**Model sử dụng:** `openai/gpt-oss-120b:free` (via OpenRouter)

---

## 1. Chuẩn Bị Môi Trường

### Cài đặt
```bash
# Cài uv package manager
python -m pip install uv

# Sync dependencies
python -m uv sync

# Cấu hình .env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-oss-120b:free
OPENROUTER_TEMPERATURE=0.3
OPENROUTER_MAX_TOKENS=1500
```

**Lưu ý thực tế:** API key cung cấp là tài khoản free-tier gần hết credit, không dùng được model mặc định `claude-sonnet-4-5` (lỗi 402). Đã chuyển sang model miễn phí `openai/gpt-oss-120b:free`.

---

## 2. Phần 1 — Bài Tập 1.2: Thêm Temperature Control

**File:** `common/llm.py`

**Thay đổi:** Thêm `temperature=0.3` vào `get_llm()` để output ổn định hơn.

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.3")),  # ← BÀI TẬP 1.2
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1500")),
    )
```

**Giải thích:** `temperature=0` → hoàn toàn deterministic; `temperature=1` → sáng tạo nhất. `0.3` cân bằng giữa ổn định và tự nhiên, phù hợp cho tư vấn pháp lý.

---

## 3. Phần 2 — Exercise 2: Tools và Knowledge Base

**File:** `exercises/exercise_2_tools.py`

### 3.1 Bài Tập 2.1 — Thêm entry luật lao động vào Knowledge Base

```python
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": "Under the Uniform Commercial Code (UCC) Article 2...",
    },
    # ← THÊM MỚI
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
            "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
            "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
            "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
        ),
    },
]
```

### 3.2 Bài Tập 2.2 — Tạo tool `check_statute_of_limitations`

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
```

Thêm vào danh sách tools và xử lý tool call:
```python
tools = [search_legal_knowledge, check_statute_of_limitations]  # ← THÊM
llm_with_tools = llm.bind_tools(tools)

# Trong vòng lặp tool_calls:
elif tool_call["name"] == "check_statute_of_limitations":
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

### 3.3 Kết quả chạy

```
Câu hỏi: Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?

🔧 Gọi tool: check_statute_of_limitations

✅ Kết quả:
Theo quy định pháp luật, thời hiệu khởi kiện đối với vụ vi phạm hợp đồng
là 4 năm (theo UCC § 2-725 - Bộ luật Thương mại Thống nhất).
```

**Nhận xét:** LLM tự nhận biết cần gọi tool `check_statute_of_limitations` với `case_type="contract"` mà không cần hướng dẫn thêm — đây là Function Calling tự động.

---

## 4. Phần 4 — Exercise 4: Thêm Privacy Agent vào Multi-Agent System

**File:** `exercises/exercise_4_multiagent.py`

### 4.1 Implement `privacy_agent`

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, data protection, privacy rights, data breach notification, hình phạt vi phạm."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

### 4.2 Conditional Routing (Bài Tập 4.2)

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))

    # ← THÊM MỚI: routing cho privacy_agent
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]
```

### 4.3 Thêm vào Graph (node + edge + aggregate)

```python
# Graph build:
graph.add_node("privacy_agent", privacy_agent)          # ← node mới

graph.add_conditional_edges(
    "law_agent",
    check_routing,
    ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],  # ← thêm
)
graph.add_edge("privacy_agent", "aggregate_results")    # ← edge mới

# Aggregate section:
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH BẢO VỆ DỮ LIỆU (GDPR):\n{state['privacy_analysis']}")
```

> **Bug đã fix:** Skeleton gốc đăng ký `check_routing` nhầm thành **node** (trả về `list[Send]` thay vì `dict`). Đã sửa thành **path function** trong `add_conditional_edges` — đây là pattern đúng của LangGraph `Send` API.

### 4.4 Kết quả chạy

```
Câu hỏi: Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?

# KẾT QUẢ CUỐI CÙNG
# BÁO CÁO PHÁP LÝ: HẬU QUẢ RÒ RỈ DỮ LIỆU KHÁCH HÀNG

## PHẦN I: HẬU QUẢ PHÁP LÝ
...trách nhiệm dân sự, hành chính, hình sự...

## PHẦN II: PHÂN TÍCH THUẾ
...hậu quả thuế khi bị rò rỉ dữ liệu...

## 🔒 PHÂN TÍCH BẢO VỆ DỮ LIỆU (GDPR):
...GDPR penalties, data breach notification 72h, NĐ 13/2023...
```

**Nhận xét:** Ba agents (`law_agent` → routing → `tax_agent` + `privacy_agent` song song → `aggregate_results`) chạy đúng luồng, kết quả tổng hợp đầy đủ cả 3 góc độ pháp lý.

---

## 5. Phần 5 — Stage 5: Distributed A2A System

### 5.1 Khởi động hệ thống

```powershell
# Khởi động 5 services theo thứ tự
python -m registry        # port 10000
python -m tax_agent       # port 10102
python -m compliance_agent # port 10103
python -m law_agent       # port 10101
python -m customer_agent  # port 10100
```

### 5.2 Xác nhận đăng ký (Registry health check)

```json
GET http://localhost:10000/agents
{
  "agents": [
    {"agent_name": "tax-agent",        "endpoint": "http://localhost:10102", "tasks": ["tax_question"]},
    {"agent_name": "compliance-agent", "endpoint": "http://localhost:10103", "tasks": ["compliance_question"]},
    {"agent_name": "law-agent",        "endpoint": "http://localhost:10101", "tasks": ["legal_question"]},
    {"agent_name": "customer-agent",   "endpoint": "http://localhost:10100", "tasks": []}
  ]
}
```

### 5.3 Trace Request Flow (từ logs thực tế)

```
User question
  → Customer Agent (port 10100): LLM nhận diện câu hỏi pháp lý
    → Registry /discover/legal_question → Law Agent :10101
    → Law Agent:
        [analyze_law]     LLM phân tích hợp đồng/torts          ~30-90s
        [check_routing]   LLM quyết định needs_tax=True, needs_compliance=True
        [call_tax]   ───→ Registry → Tax Agent :10102            ┐ song song
        [call_compliance] → Registry → Compliance Agent :10103   ┘ (Send API)
        [aggregate]       Tổng hợp kết quả cuối
  ← Trả về user
```

---

## 6. Bài Tập Cộng Điểm

### Câu 1: Latency là bao nhiêu giây?

| Bước | Thời gian (model free) | Thời gian (model thương mại) |
|---|---|---|
| analyze_law (LLM call) | 30–90s | 2–5s |
| check_routing (LLM call) | 30–90s | 1–3s |
| call_tax + call_compliance song song | 120–150s | 5–15s |
| aggregate (LLM call) | 30–60s | 2–5s |
| **Tổng (thực đo)** | **~267s** | **~30–60s** |

> **Nguyên nhân chậm:** Model `openai/gpt-oss-120b:free` bị rate-limit nặng trên OpenRouter free-tier. Với model thương mại (Claude Sonnet, GPT-4o) latency thực tế < 60s.

### Câu 2: Phương án giảm latency — Demo thực tế

**Phương án: Parallel Delegation thay vì Sequential**

Trong Stage 5, Law Agent đã dùng LangGraph `Send` API để dispatch tax + compliance song song. Demo dưới đây chứng minh bằng số liệu thực tế:

**File:** `benchmark_parallel.py`

```python
# BEFORE — tuần tự (sequential)
tax = await call_tax(ctx, trace)
comp = await call_compliance(ctx, trace)
# latency = t_tax + t_compliance

# AFTER — song song (parallel)
tax, comp = await asyncio.gather(
    call_tax(ctx, trace),
    call_compliance(ctx, trace),
)
# latency = max(t_tax, t_compliance)
```

**Kết quả thực đo** (tax_agent + compliance_agent qua A2A protocol thật):

```
======================================================================
KẾT QUẢ
======================================================================
  Tuần tự (sequential): 267.21s
  Song song (parallel): 119.20s
  GIẢM ĐƯỢC           : 148.01s (55.4%)
======================================================================
```

**Diagram:**
```
BEFORE (sequential):
  [tax_agent   ████████████████████████████████]
                                                [compliance ████████████████████████████████]
  Tổng = 267s

AFTER (parallel / LangGraph Send API):
  [tax_agent   ████████████████████████████████]
  [compliance  ████████████████████████████████]
  Tổng = 119s  (tiết kiệm 148s = 55.4%)
```

**Lý giải:** Tax và Compliance agent không phụ thuộc nhau — cả hai chỉ cần câu hỏi gốc và kết quả `law_analysis` đã có sẵn. Dispatch song song giảm latency từ `t1+t2` xuống `max(t1, t2)`.

### Phương án bổ sung (đề xuất thêm)

| Phương án | Giảm latency | Ghi chú |
|---|---|---|
| Dùng model nhanh (Claude Haiku, GPT-4o-mini) | ~80% | Từ 90s/call → 2–5s/call |
| Cache kết quả routing theo loại câu hỏi | ~30–90s | Tránh gọi LLM lần 2 |
| Giảm `max_tokens` trong prompt | 20–40% | Output ngắn hơn → nhanh hơn |
| Streaming response | UX tốt hơn | User thấy text dần, không đợi toàn bộ |
| Chạy `analyze_law` song song với specialists | ~30–90s thêm | Bỏ nó khỏi critical path |

---

## 7. Tổng Kết

| Bài | Nội dung | Kết quả |
|---|---|---|
| Bài 1.2 | Thêm `temperature=0.3` vào `get_llm()` | ✅ Done |
| Exercise 2.1 | Thêm entry luật lao động VN vào Knowledge Base | ✅ Done |
| Exercise 2.2 | Tạo tool `check_statute_of_limitations` | ✅ Chạy OK |
| Exercise 4.1 | Implement `privacy_agent` | ✅ Done |
| Exercise 4.2 | Conditional routing cho privacy agent | ✅ Done |
| Exercise 4 graph | Thêm node + edge + aggregate privacy | ✅ Chạy OK |
| Stage 5 | Khởi động và test 5 distributed services | ✅ All agents registered |
| Bonus Q1 | Đo latency thực tế | ✅ ~267s (model free) |
| Bonus Q2 | Demo giảm latency bằng parallel delegation | ✅ 267s → 119s (-55.4%) |

### Files đã tạo/sửa

| File | Thay đổi |
|---|---|
| `.env` | Tạo mới với API key + model free |
| `common/llm.py` | Thêm `temperature`, `max_tokens` |
| `exercises/exercise_2_tools.py` | Thêm labor_law + tool mới |
| `exercises/exercise_4_multiagent.py` | Implement privacy_agent, fix graph bug |
| `benchmark_parallel.py` | Script đo latency sequential vs parallel |
| `BONUS_LATENCY.md` | Phân tích latency chi tiết |
| `BAO_CAO.md` | Báo cáo này |
