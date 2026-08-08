import asyncio
import time

import httpx

ENDPOINT = "http://localhost:8000/v1/chat/completions"
PROMPTS = 500
CONCURRENCY = 4


async def one(client: httpx.AsyncClient, i: int) -> float:
    started = time.monotonic()
    await client.post(
        ENDPOINT,
        json={
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": f"Hello #{i}, tell me about decentralized AI in 50 words."}],
            "max_tokens": 64,
        },
    )
    return time.monotonic() - started


async def main() -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        latencies = await asyncio.gather(*(one(client, i) for i in range(PROMPTS)))
    latencies.sort()
    n = len(latencies)
    p50, p95, p99 = latencies[n // 2], latencies[int(n * 0.95)], latencies[int(n * 0.99)]
    total = sum(latencies)
    print(f"requests: {n} | total {total:.1f}s | throughput {n / max(total, 0.001):.1f} req/s")
    print(f"p50 {p50:.2f}s | p95 {p95:.2f}s | p99 {p99:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())