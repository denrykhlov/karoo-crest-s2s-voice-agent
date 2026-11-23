# Karoo Crest Bank — Multilingual Pure-S2S Voice Agent + Flametree Delegation

A real-time, **pure speech-to-speech (zero external VAD)** multilingual voice agent powered by **`LiquidAI/LFM3-Omni-2B`**, running on either:
- **macOS Apple Silicon** via native **MLX** (`--backend mlx`, with `B=2` heterogeneous batching + Frame-1 `AudioWorklet` streaming), or
- **Linux / NVIDIA CUDA (Runpod)** via **`vllm-omni`** (`--backend vllm`, continuous batching + `--async-chunk` codec streaming + `/v1/chat/completions` & `/v1/audio/speech`),

with per-message **TTFT** / **TTFA** telemetry, per-call artifact folders, and live **Flametree Platform** knowledge delegation.

---

## 1. Key Capabilities

### Multilingual Pure-S2S (`es-MX` · `es` · `en`)
- Full browser-to-S2S speech understanding and 24 kHz voice generation in **Mexican Spanish (`es-MX`)**, **Spanish (`es`)**, and **English (`en`)** — selectable on the fly via CLI (`--lang`) or live mid-call in the Call Studio UI.
- Unicode-aware Spanish/English semantic turn-end detection (`¿...?`, Spanish conjunction continuation cues) and barge-in exempt phrase handling (`espera`, `alto`, `disculpa`, `wait`, `stop`, `sorry`).

### Low-Latency Streamed Audio & Per-Message Metrics
- **Frame-1 / Async-Chunk Speech Streaming**: Immediately pushes synthesized 24 kHz PCM16 audio chunks over WebSocket as soon as the first codec chunk is decoded, reporting browser-side `playback_started` back to compute **Perceived TTFA**.
- **Per-Generated-Message Latency Chips**: Every assistant message renders live **`TTFT`**, **`TTFA`**, and **`Perceived`** timings in the transcript feed and stores them per message in `<calls_dir>/<call_id>/transcript.json` and inline in `transcript.txt` / exported `.txt`.
- **Call-Aggregate Delivery Telemetry (`voice_agent/metrics.py`)**: Per-turn and call-aggregate (`last` / `avg` / `p50` / `p95`) metrics streamed live to **Delivery Stats**.

### Parallel Knowledge Delegation + Dynamic Fillers
- **`mlx` backend**: Uses MLX's `BatchKVCache` / `BatchRotatingKVCache` (`B=2` heterogeneous batching) to run voice fillers (`always_delegate`) or `DIRECT`/`DELEGATE:` routing (`selective_tool`) in parallel within one decode step.
- **`vllm` backend**: Runs concurrent OpenAI-compatible HTTP/SSE streams against `vllm serve --omni` continuous batching (no `mlx` dependency on Linux/CUDA), stripping WAV headers cleanly and falling back to `/v1/audio/speech` if text arrives without audio.
- Supports **live mid-call switching** between `selective_tool` and `always_delegate`, plus toggling parallel fillers (`--no-filler` / UI checkbox).

### Per-Call Artifact Folders (`voice_agent/call_recorder.py`)
Every call receives a unique UTC-sortable `call_id` (`call-YYYYMMDD-HHMMSS-<8hex>`) and writes incremental + finalized artifacts into `<calls_dir>/<call_id>/` (default `./calls/<call_id>/`, configurable via `--calls-dir` or `CALLS_DIR`, disable WAV capture with `--no-record-audio`):
- `metadata.json`: UTC timestamps, duration, backend, settings & mid-call config changes, Flametree session ID/URL (tokens redacted), audio durations, and counters.
- `trace.jsonl`: timestamped event log (`call_started`, `first_token_sent`, `first_audio_chunk_sent`, speech start/delta/commit, barge-ins, router/filler/query events, Flametree calls, CRM tools, `call_ended`).
- `transcript.txt` & `transcript.json`: dialogue transcript with timestamps, speaker labels, interruption markers, and **per-message `metrics` (`ttft_ms`, `ttfa_server_ms`, `ttfa_perceived_ms`)**.
- `metrics.json`: call-level aggregates and every turn's delivery metrics.
- `crm_and_delegation.json`: CRM tool state & log plus all Flametree delegation queries/responses.
- `audio/caller_16k.wav` & `audio/agent_24k.wav`: optional mono 16-bit PCM recordings.
- Read-only inspection routes: `GET /api/calls` and `GET /api/calls/{call_id}`.

---

## 2. Flametree Platform Delegation

Complex banking knowledge (tariffs, fees, 55-day grace rules, NCA affordability multipliers, FICA documents, POPIA consent, fraud & travel perks) is hosted on the **Flametree Platform (`prod` · tenant `Denis Demo`)**:
- **Agent**: `Karoo Crest Bank Knowledge Expert (S2S Test)` (`1717ee86-0823-4700-aa70-d2b2b1c165bb`)
- **Portal URL**: `https://portal.flametree.ai/agent-v2/1717ee86-0823-4700-aa70-d2b2b1c165bb?tenant_id=8470d927-5847-48fc-9a55-d72dd4117ac1`

Set the agent's Chat API bearer token in your environment to enable live Flametree sessions (`POST /chatbot/{agent_id}/session`, multipart `run_v2`, `DELETE` on hang-up). When unset, `LocalKnowledgeFallbackDelegate` serves offline fallback answers so local testing works out of the box:

```bash
export FLAMETREE_AGENT_TOKEN="<your-agent-token>"
```

---

## 3. Project Layout

- [`telesales_agent.py`](telesales_agent.py): Entrypoint, personas (**Naledi Mokoena** / **Thabo Dlamini**), 4 CRM tools, `--backend mlx|vllm`, `--vllm-deploy-config`, multilingual & Flametree wiring.
- [`voice_agent/protocol.py`](voice_agent/protocol.py): Backend-agnostic shared `TurnRecord`, `TurnOutcome`, `BatchPlanOutcome`, speech/echo/endpointing helpers, and WAV/tool helpers (zero `mlx` imports).
- [`voice_agent/backend_factory.py`](voice_agent/backend_factory.py): Lazy backend dispatcher (`mlx` -> `Lfm3OmniEngine`, `vllm` -> `VllmOmniEngine`).
- [`voice_agent/vllm_backend.py`](voice_agent/vllm_backend.py): Pure-S2S native in-process `vllm_omni.AsyncOmni` backend (`VllmOmniEngine`, `VllmLiveS2SListener`, `VllmStreamingConversationSession`).
- [`voice_agent/engine.py`](voice_agent/engine.py), [`voice_agent/live_pipeline.py`](voice_agent/live_pipeline.py), [`voice_agent/batch_gen.py`](voice_agent/batch_gen.py), [`voice_agent/codec_stream.py`](voice_agent/codec_stream.py): Native macOS MLX `B=2` engine & LiquidCodec streamer.
- [`voice_agent/languages.py`](voice_agent/languages.py): `es-MX`, `es`, `en` language profiles, prompts, filler templates, and endpointing cue sets.
- [`voice_agent/delegation.py`](voice_agent/delegation.py): `FlametreeHttpDelegate` and `LocalKnowledgeFallbackDelegate`.
- [`voice_agent/metrics.py`](voice_agent/metrics.py): `DeliveryStatsTracker` & `TurnDeliveryMetrics`.
- [`voice_agent/call_recorder.py`](voice_agent/call_recorder.py): `CallRecorder` & inspection helpers for per-call artifact folders.
- [`voice_agent/session.py`](voice_agent/session.py) & [`voice_agent/server.py`](voice_agent/server.py): FastAPI + WebSocket orchestrator.
- [`voice_agent/static/`](voice_agent/static/): Browser Call Studio UI with 40 ms prebuffer `AudioWorklet`, per-message TTFT/TTFA chips, Delivery Stats, and CRM cards.
- [`deploy/lfm3_omni_runpod.yaml`](deploy/lfm3_omni_runpod.yaml), [`deploy/start_runpod.sh`](deploy/start_runpod.sh), [`Dockerfile.vllm`](Dockerfile.vllm), [`.github/workflows/build-runpod-vllm.yml`](.github/workflows/build-runpod-vllm.yml): CUDA 13 `vllm-omni` container build & Runpod launcher.

---

## 4. Setup & Running

### macOS (MLX Backend)
```bash
uv sync
export FLAMETREE_AGENT_TOKEN="<optional-for-live-flametree>"
./run.sh --open

# Or customize backend/language/mode:
uv run python telesales_agent.py --backend mlx --lang es-MX --delegation-mode selective_tool --voice naledi --patience balanced --open
```

### Linux / In-Process `vllm-omni` (`AsyncOmni`)
```bash
python3 telesales_agent.py --backend vllm \
  --model LiquidAI/LFM3-Omni-2B \
  --model-revision 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67 \
  --vllm-deploy-config deploy/lfm3_omni_runpod.yaml \
  --host 0.0.0.0 --port 8000
```

### Runpod GPU Pod (GHCR Container)
1. Push to `main` to build `linux/amd64` via GitHub Actions and publish `ghcr.io/denrykhlov/karoo-crest-s2s-voice-agent:v0.1.4` (pinned to CUDA 13 + `vllm/vllm-openai:v0.33.0` + `vllm-omni@ca9398dc`, running single-process `AsyncOmni` directly in-process).
2. Deploy a Runpod Pod on **CUDA >= 13.0** (e.g., `NVIDIA A40` 48 GB or `NVIDIA L4` / `RTX 4090` 24 GB) exposing `8000/http` with a persistent `/workspace` volume for HuggingFace model weights (`/workspace/.cache/huggingface`) and call artifacts (`/workspace/calls`).
3. Optional environment variables on the Pod:
   - `MODEL_ID` (default `LiquidAI/LFM3-Omni-2B`)
   - `MODEL_REVISION` (default `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`)
   - `FLAMETREE_AGENT_TOKEN` (set via Runpod Secrets/env if live Flametree knowledge delegation is desired)
   - `DEFAULT_LANG` (`en`, `es-MX`, `es`), `DEFAULT_VOICE` (`naledi`, `thabo`), `DELEGATION_MODE` (`selective_tool`, `always_delegate`)

---

## 5. Running Tests

```bash
node --check voice_agent/static/app.js
uv run pytest -v -s
```
