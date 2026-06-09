# Bài Tập Cộng Điểm — Latency & Tối Ưu Hóa

## Câu hỏi 1: Latency là bao nhiêu giây?

**Đo trên Stage 5 distributed A2A, với model `openai/gpt-oss-120b:free`:**

| Bước | Thời gian |
|---|---|
| Customer Agent nhận request → gọi Law Agent (A2A) | ~1–2s |
| Law Agent: `analyze_law` (LLM call) | ~30–90s |
| Law Agent: `check_routing` (LLM call) | ~30–90s |
| Law Agent: `call_tax` + `call_compliance` **song song** (A2A) | ~120–150s |
| Law Agent: `aggregate` (LLM call) | ~30–60s |
| **Tổng cộng (sequential hết)** | **~267s (~4.5 phút)** |

> **Gốc rễ vấn đề:** Model free trên OpenRouter bị rate-limit nặng (mỗi lượt gọi
> LLM mất 30–90s do throttle). Với model thương mại (Claude Sonnet, GPT-4o) thì
> latency thực tế khoảng **30–60s** cho toàn bộ chain.

---

## Câu hỏi 2: Đề xuất phương án giảm latency

### Phương án đã demo: Parallel Delegation (asyncio.gather / Send API)

**Nguyên lý:**
- BEFORE: gọi Tax Agent → *chờ xong* → gọi Compliance Agent (tuần tự)
- AFTER: gọi Tax Agent **và** Compliance Agent **cùng lúc** (song song)

```
BEFORE (sequential):
  [tax_agent   ████████████████████]
                                    [compliance  ██████████████████████]
  latency = t_tax + t_compliance = 267s

AFTER (parallel):
  [tax_agent   ████████████████████]
  [compliance  ██████████████████████]
  latency = max(t_tax, t_compliance) = 119s
```

**Kết quả thực đo:**

| | Sequential | Parallel | Giảm |
|---|---|---|---|
| Latency | 267.21s | 119.20s | **148s (55.4%)** |

Code thực hiện (`asyncio.gather` — tương đương LangGraph `Send` API):

```python
# BEFORE — tuần tự
tax = await call_tax(ctx, trace)
comp = await call_compliance(ctx, trace)
# latency = t_tax + t_compliance

# AFTER — song song
tax, comp = await asyncio.gather(
    call_tax(ctx, trace),
    call_compliance(ctx, trace),
)
# latency = max(t_tax, t_compliance)
```

**Trong Stage 5 (distributed):** Law Agent đã dùng LangGraph `Send` API để
dispatch `call_tax` và `call_compliance` song song (xem `law_agent/graph.py` →
`route_to_subagents`). Đây chính là lý do Stage 4/5 nhanh hơn kiến trúc sequential
đơn thuần.

---

### Phương án bổ sung (không cần code thêm)

| Phương án | Giảm latency ước tính | Ghi chú |
|---|---|---|
| **Model nhanh hơn** (GPT-4o-mini, Claude Haiku) | 70–80% | Mỗi LLM call từ 90s → 2–5s |
| **Caching routing decision** | ~30–90s | Câu hỏi cùng loại → bỏ `check_routing` call |
| **Prompt ngắn hơn** (giảm max_tokens) | 20–40% | Ít token output → nhanh hơn |
| **Streaming response** | UX cải thiện | User thấy output dần, không chờ toàn bộ |
| **Chạy `analyze_law` song song với specialists** | ~30–90s thêm | Bỏ nó khỏi critical path |

---

## Tóm tắt

- **Latency gốc (Stage 5, sequential fallback):** ~267s (do model free chậm)  
- **Sau tối ưu parallel delegation:** ~119s → **giảm 55.4%**
- **Bottleneck thực tế:** thời gian LLM inference (model free bị throttle nặng)
- **Giải pháp production:** dùng model thương mại + parallel Send API → latency < 30s
