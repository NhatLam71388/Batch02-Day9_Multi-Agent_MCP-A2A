"""Bài Tập Cộng Điểm — Demo tối ưu latency bằng PARALLEL DELEGATION.

Đây là bài học cốt lõi của Stage 4/5: dispatch các specialist agent SONG SONG
(LangGraph Send API / asyncio.gather) thay vì gọi TUẦN TỰ.

So sánh cùng model, cùng câu hỏi, cùng 2 agent (tax + compliance):

  [BEFORE] Tuần tự:  await tax  -> rồi await compliance
           latency = t_tax + t_compliance

  [AFTER]  Song song: await asyncio.gather(tax, compliance)
           latency = max(t_tax, t_compliance)

Yêu cầu: registry + tax_agent + compliance_agent đang chạy.

Chạy:
    uv run python benchmark_parallel.py
"""

import asyncio
import time
from uuid import uuid4

from dotenv import load_dotenv

from common.a2a_client import delegate
from common.registry_client import discover

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


async def call_tax(ctx, trace):
    endpoint = await discover("tax_question")
    return await delegate(endpoint=endpoint, question=QUESTION, context_id=ctx, trace_id=trace, depth=1)


async def call_compliance(ctx, trace):
    endpoint = await discover("compliance_question")
    return await delegate(endpoint=endpoint, question=QUESTION, context_id=ctx, trace_id=trace, depth=1)


async def run_sequential() -> float:
    ctx, trace = str(uuid4()), str(uuid4())
    t0 = time.perf_counter()
    tax = await call_tax(ctx, trace)
    comp = await call_compliance(ctx, trace)
    dt = time.perf_counter() - t0
    print(f"  [SEQUENTIAL] tax={len(tax)} chars, compliance={len(comp)} chars -> {dt:.2f}s")
    return dt


async def run_parallel() -> float:
    ctx, trace = str(uuid4()), str(uuid4())
    t0 = time.perf_counter()
    tax, comp = await asyncio.gather(call_tax(ctx, trace), call_compliance(ctx, trace))
    dt = time.perf_counter() - t0
    print(f"  [PARALLEL]   tax={len(tax)} chars, compliance={len(comp)} chars -> {dt:.2f}s")
    return dt


async def main():
    load_dotenv()
    print("=" * 70)
    print("DEMO TỐI ƯU LATENCY: SEQUENTIAL vs PARALLEL DELEGATION")
    print("=" * 70)
    print(f"Câu hỏi: {QUESTION}\n")

    print("Chạy TUẦN TỰ (gọi tax xong mới gọi compliance)...")
    seq = await run_sequential()

    print("\nChạy SONG SONG (asyncio.gather - giống Send API trong LangGraph)...")
    par = await run_parallel()

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"  Tuần tự (sequential): {seq:.2f}s")
    print(f"  Song song (parallel): {par:.2f}s")
    if seq > 0:
        saved = seq - par
        pct = saved / seq * 100
        print(f"  GIẢM ĐƯỢC           : {saved:.2f}s ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
