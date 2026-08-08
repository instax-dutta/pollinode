# PolliNode

Open-source agent inference routing for decentralized GPU networks.

Loads an OpenAI-compatible `/v1/chat/completions` endpoint and routes each request to the best available provider - starting with the Nosana GPU marketplace - using live load, price, and rolling-latency scores. Provider down or over budget? Requests fail over to the next pool automatically.

No single API chokepoint. No vendor lock-in. Per-request transparency about where your tokens actually ran.

## Status: architecture build - Decentralize AI Hackathon, Round 1

- [x] Repo + pollinode skeleton
- [x] OpenAI-compatible chat completions shim
- [x] Provider registry with scoring + failover
- [x] Budget caps per provider
- [x] Dockerfile for router + vLLM worker
- [x] Nosana deploy config (`deploy.yaml`, Simple strategy)
- [x] Benchmark harness (latency / throughput / uptime)
- [ ] Live Nosana deployment of Llama 3.1 8B and verified numbers
- [ ] Fine-tune track (Round 2: LoRA on Qwen2.5-7B)
- [ ] Round 1 HackerNoon writeup

## Architecture

```
client ──> /v1/chat/completions (FastAPI router)
                │
                ├── score_providers()   # price + rolling latency + health
                ├── pick_best()         # top scored within budget caps
                └── failover()          # on error/limit, next provider
                                           │
                              ┌────────────┴────────────┐
                              ▼                          ▼
                    Nosana vLLM worker        (future) other GPU pools
                    (Llama/Qwen, 8B-14B)      (Akash, io.net, ...)
```

## Quick start (local dev)

```bash
docker compose up                     # router + local vLLM worker
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"hi"}]}'
```

## Deploy on Nosana

```bash
# configure providers/api keys
cp .env.example .env

# deploy.yaml uses Simple strategy, 1 replica, 2h container timeout
# (see deploy.yaml - adjust GPU market to fit your budget)
```

## Benchmarks

```bash
python benchmark/bench.py --endpoint http://localhost:8000/v1/chat/completions \
  --prompts 100 --concurrency 4
```

## Roadmap

- [ ] Live Nosana deployment + benchmark vs centralized API (writeup)
- [ ] vLLM worker image with auto-configured model
- [ ] LoRA fine-tune track
- [ ] Multi-pool orchestration (Nosana + others)

Built for the [Decentralize AI Hackathon](https://decentralizeai.tech) - the AI future should be open, transparent, and owned by its users. MIT licensed.