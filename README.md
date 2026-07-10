# Tetrlense ai — Four Voices, One Vision

**Style-Conditioned Video Captioning Agent built for AMD Developer Hackathon: ACT II (Track 2)**

*   **Developer:** Alagappan
*   **Team:** VectorForge AI
*   **Core Architecture:** Two-Pass Factual Grounding (Scene Understanding → Styled Captioning) using Moonshot's Kimi K2 vision model via Fireworks AI.

---

## How It Works

```
tasks.json --> Download Video --> Extract Frames --> Pass 1: Scene Understanding --> Pass 2: Styled Captioning --> results.json
```

### Two-Pass Architecture

1. **Pass 1 — Scene Understanding**: Sends sampled frames to Kimi K2 and gets a factual, objective description of what's happening in the video. This creates a "grounding document" that prevents hallucination.

2. **Pass 2 — Styled Captioning**: Using the scene description as context plus the original frames, generates all four styled captions in a single call. Each style has a deeply characterized persona with explicit DO/DON'T constraints and reference examples.

This separation ensures **accuracy first, tone second** — exactly what the LLM-Judge evaluates.

### Key Design Decisions

- **Kimi K2 with thinking disabled**: Kimi K2 is a reasoning model. By default it generates large amounts of internal "thinking" text before its answer. Passing `"thinking": {"type": "disabled"}` skips this step, dropping per-clip latency from ~37s to ~7s — well under the 30-second-per-request contest limit.

- **Scene-change frame extraction**: Instead of sampling frames at fixed intervals, the agent first tries ffmpeg's scene detection filter (`select='gt(scene,0.25)'`) to grab frames at actual visual transitions. Falls back to uniform sampling if scene detection yields too few or too many frames.

- **Per-style retry**: If any style fails to parse from the JSON response, the agent retries just that style individually with a focused single-style prompt, rather than re-running the entire batch.

- **Parallel video processing**: Videos are processed concurrently using a thread pool (3 workers), significantly reducing total pipeline time for larger batches.

- **Robust JSON parsing**: Handles markdown fences, stray text, partial JSON, and falls back to regex key-value extraction as a last resort.

## Prerequisites

- Docker with buildx (for cross-platform builds)
- A Fireworks AI API key

## Quick Start

### 1. Test locally (without Docker)

```bash
pip install -r requirements.txt
# ffmpeg must be installed:
#   Windows: winget install ffmpeg
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt-get install ffmpeg

export FIREWORKS_API_KEY=your_key_here
export TASKS_PATH=./test_input/TASKS.json
export RESULTS_PATH=./test_output/results.json
python app.py
cat test_output/results.json
```

### 2. Run the Streamlit Interactive Dashboard (for presenting/testing)

We built an interactive, dark-themed Streamlit dashboard (`demo_app.py`) so you can run the pipeline live on any video and inspect frame extraction, logs, and caption cards side-by-side:

```bash
pip install -r requirements_demo.txt
export FIREWORKS_API_KEY=your_key_here
streamlit run demo_app.py
```

### 3. Build the Docker image (linux/amd64)

```bash
docker buildx build --platform linux/amd64 -t video-captioning-agent:latest --load .
```

### 3. Run the container

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=your_key_here \
  -v "$(pwd)/test_input:/input" \
  -v "$(pwd)/test_output:/output" \
  video-captioning-agent:latest
```

### 4. Verify output

```bash
cat test_output/results.json
```

Check that:
- Exit code is 0
- `results.json` is valid JSON
- Every task has a caption for every requested style
- Finishes well under the 10-minute runtime limit

## Push to Public Registry

```bash
docker tag video-captioning-agent:latest ghcr.io/<your-username>/video-captioning-agent:latest
docker buildx build --platform linux/amd64 -t ghcr.io/<your-username>/video-captioning-agent:latest --push .
```

Make sure the image is **public** before submitting.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREWORKS_API_KEY` | Yes | — | Your Fireworks AI API key |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | Fireworks API endpoint |
| `FIREWORKS_MODEL` | No | `accounts/fireworks/models/kimi-k2p6` | Vision model to use |
| `TASKS_PATH` | No | `/input/tasks.json` | Path to input tasks file |
| `RESULTS_PATH` | No | `/output/results.json` | Path to output results file |
| `NUM_FRAMES` | No | `5` | Frames sampled per video clip |
| `MAX_FRAME_DIM` | No | `768` | Max pixel dimension for frame resizing |
| `TWO_PASS` | No | `1` | Set to `0` to disable two-pass architecture |

## Project Structure

```
.
├── app.py              # Main captioning agent
├── evaluate.py         # Self-evaluation script (LLM-judge simulator)
├── requirements.txt    # Python dependencies
├── dockerfile          # Container definition
├── test_input/
│   └── TASKS.json      # Example test tasks
└── test_output/
    └── results.json    # Generated captions
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Captioning Agent                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  tasks.json ──> Download ──> Frame Extraction                   │
│                              (scene-change or uniform)          │
│                                    │                            │
│                                    v                            │
│                          ┌─────────────────┐                    │
│                          │  Pass 1: Scene   │                   │
│                          │  Understanding   │                   │
│                          │  (Kimi K2, t=0.3)│                   │
│                          └────────┬────────┘                    │
│                                   │ factual description         │
│                                   v                             │
│                          ┌─────────────────┐                    │
│                          │  Pass 2: Styled  │                   │
│                          │  Captioning      │                   │
│                          │  (Kimi K2, t=0.7)│                   │
│                          └────────┬────────┘                    │
│                                   │ 4 styled captions           │
│                                   v                             │
│                 ┌─────────────────────────────────┐             │
│                 │  Validate + Per-Style Retry      │             │
│                 └────────────────┬────────────────┘             │
│                                  │                              │
│                                  v                              │
│                           results.json                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scoring Notes

Track 2 is scored via **LLM-Judge on two axes**:
- **Accuracy**: How well the caption reflects what actually happens in the video
- **Tone adherence**: How well each caption matches its requested style

The two-pass architecture is specifically designed to maximize both:
- Pass 1 ensures factual grounding (accuracy)
- Pass 2's detailed personas with DO/DON'T constraints ensure style separation (tone)

## Tech Stack

- **Model**: Kimi K2 (`kimi-k2p6`) via Fireworks AI API
- **Video Processing**: FFmpeg (frame extraction, scene detection, resizing)
- **Runtime**: Python 3.11, Docker (linux/amd64)
- **Cloud**: AMD Developer Cloud + Fireworks AI API credits

## License

MIT
