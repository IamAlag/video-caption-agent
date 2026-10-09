# Tetrlense AI — Four Voices, One Vision

A **style-conditioned video captioning prototype** built for the AMD Developer Hackathon: ACT II (Track 2). It uses a two-pass vision-language workflow to generate captions in multiple styles while keeping factual scene understanding separate from creative wording.

- **Developer:** Alagappan
- **Team:** VectorForge AI
- **Core workflow:** scene understanding → style-conditioned caption generation
- **Model provider:** Fireworks AI (Kimi vision model; configurable via environment variables)

## The problem

A caption can sound polished while describing something that never happened in the video. This project experiments with separating the factual description of a clip from the later task of writing in a particular voice.

## How it works

```mermaid
flowchart TD
    A[Task list / video URLs] --> B[Download or load clips]
    B --> C[Extract representative frames]
    C --> D[Pass 1: factual scene description]
    D --> E[Pass 2: captions in requested styles]
    E --> F[Parse and validate JSON]
    F --> G{All styles valid?}
    G -- No --> H[Retry only failed styles]
    H --> F
    G -- Yes --> I[Write results.json]
```

### Key engineering decisions

- **Two-pass generation:** first create a factual grounding description, then use it as context for styled captions. This is intended to reduce unsupported details; it does not eliminate hallucinations.
- **Scene-aware frame extraction:** FFmpeg scene detection is attempted first, with uniform sampling as a fallback when the number of selected frames is unsuitable.
- **Targeted retries:** retry an individual style if its output cannot be parsed, instead of rerunning every style.
- **Concurrent processing:** a thread pool processes multiple clips concurrently (configured for three workers).
- **Defensive output parsing:** handles common formatting problems such as Markdown fences and surrounding text, with fallback parsing for imperfect model output.

## Tech stack

Python · FFmpeg · Fireworks AI API · Docker · Streamlit

## Quick start

### Requirements

- Python
- FFmpeg
- A Fireworks AI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the API key and input/output paths:

```bash
export FIREWORKS_API_KEY=your_key_here
export TASKS_PATH=./test_input/TASKS.json
export RESULTS_PATH=./test_output/results.json
python app.py
```

Inspect the generated `results.json` and verify that each task has the expected styles. The exact input schema is defined by the sample task file in the repository.

### Run the interactive dashboard

```bash
pip install -r requirements_demo.txt
export FIREWORKS_API_KEY=your_key_here
streamlit run demo_app.py
```

### Run with Docker

Build:

```bash
docker buildx build --platform linux/amd64 -t video-captioning-agent:latest --load .
```

Run (mount local input and output directories):

```bash
docker run --rm \
  -e FIREWORKS_API_KEY \
  -v "$(pwd)/test_input:/input" \
  -v "$(pwd)/test_output:/output" \
  video-captioning-agent:latest
```

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `FIREWORKS_API_KEY` | Yes | — | Provider authentication |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | API endpoint |
| `FIREWORKS_MODEL` | No | `accounts/fireworks/models/kimi-k2p6` | Model identifier |
| `TASKS_PATH` | No | `/input/tasks.json` | Input task file |
| `RESULTS_PATH` | No | `/output/results.json` | Output JSON file |
| `NUM_FRAMES` | No | `5` | Frame sampling target |
| `MAX_FRAME_DIM` | No | `768` | Maximum frame dimension |
| `TWO_PASS` | No | `1` | Set to `0` to disable the two-pass flow |

## Limitations and next steps

- Caption quality depends on the selected model and sampled frames; events between sampled frames can be missed.
- A grounding description can itself be wrong, and the second pass can introduce details not present in the video.
- Provider latency, rate limits, and API costs affect throughput.
- The most useful next improvements are a repeatable evaluation set, tests for malformed model responses, measurements for latency/cost, and clearer failure reporting for failed downloads or clips.

## Project structure

```text
app.py                 # Main captioning pipeline
demo_app.py            # Interactive Streamlit demo
evaluate.py             # Evaluation / judge simulation
requirements.txt        # Runtime dependencies
Dockerfile              # Container image
test_input/             # Sample task input
```

## About

Built by Alagappan as a hackathon project exploring multimodal AI pipelines, structured model outputs, retries, and containerized execution. This is a prototype, not a claim of production-grade video understanding.
