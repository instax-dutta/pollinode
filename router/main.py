"""PolliNode - OpenAI-compatible inference router for decentralized GPU networks.

Routes /v1/chat/completions to the best-scoring healthy provider,
with failover and per-provider budget caps.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="PolliNode", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


DEFAULT_PROVIDERS: dict[str, dict[str, Any]] = {
    "nosana": {
        "base_url": "http://worker:8000/v1",  # local vLLM worker; swap for deployed endpoint
        "api_key": "local-demo",
        "models": ["llama-3.1-8b", "qwen2.5-7b"],
        "budget_usd": 70.0,
        "spent_usd": 0.0,
        "health": True,
        "latency_ema_s": 1.0,
    },
}


class Router:
    def __init__(self, providers: dict[str, dict[str, Any]]) -> None:
        self.providers = providers
        self.clients = {
            name: AsyncOpenAI(base_url=p["base_url"], api_key=p["api_key"])
            for name, p in providers.items()
        }
        self._lock = asyncio.Lock()
        self._mutex = asyncio.Lock()

    async def select(self, model: str, roll: float = 0.5) -> tuple[str, AsyncOpenAI]:
        """Pick best provider: score = latency_ema * (1 + roll) / budget_remaining."""
        async with self._lock:
            viable = [
                (name, p)
                for name, p in self.providers.items()
                if p["health"] and model in p["models"] and p["spent"] < p["budget"]
            ]
        if not viable:
            raise HTTPException(503, "No healthy provider within budget model")

        def score(p: dict[str, Any]) -> float:
            remaining = max(p["budget"] - p["spent"], 0.01)
            return p["latency_ema_s"] * (1 + roll) / remaining

        best_name, best = min(viable, key=lambda np: score(np[1]))
        return best_name, self.clients[best_name]

    async def record(self, name: str, latency_s: float, cost_usd: float) -> None:
        async with self._lock:
            p = self.providers[name]
            p["latency_ema_s"] = 0.9 * p["latency_ema_s"] + 0.1 * max(latency_s, 0.01)
            p["spent"] += cost_usd

    async def fail(self, name: str) -> None:
        async with self._lock:
            self.providers[name]["health"] = False


router = Router(DEFAULT_PROVIDERS)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"providers": {n: {"healthy": p["health"], "spent": p["spent"], "budget": p["budget"]} for n, p in router.providers.items()}}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    if req.stream:
        raise HTTPException(400, "streaming not implemented in this scaffold")

    last_error: Exception | None = None
    for attempt in range(len(router.providers)):
        name, client = await router.select(req.model, roll=0.5 + 0.15 * attempt)
        started = time.monotonic()
        try:
            resp = await client.chat.completions.create(
                model=req.model,
                messages=[m.model_dump() for m in req.messages],
                max_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.7,
            )
            latency = time.monotonic() - started
            est_cost = max(latency * 0.0002, 0.0004)  # crude $/s placeholder
            await router.record(name, latency, est_cost)
            return {
                "object": "chat.completion",
                "model": req.model,
                "provider": name,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(resp.choices[0].message.content.split()),
                    "total_tokens": 0,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": resp.choices[0].message.content},
                        "finish_reason": resp.choices[0].finish_reason,
                    }
                ],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await router.fail(name)

    raise HTTPException(502, f"all providers failed: {last_error}")