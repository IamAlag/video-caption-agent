# Start Here: Learn the Video Captioning Agent

This guide explains the project in everyday language, then shows where to look in the code and how to discuss the design in an interview.

## 1. What does it do?

The program takes a list of video URLs and creates four captions for each video:
- **formal** — neutral, documentary-like language
- **sarcastic** — dry, ironic humour
- **humorous_tech** — jokes based on programming and software engineering
- **humorous_non_tech** — everyday, non-technical humour

The central idea is to separate **what appears to happen in the video** from **how the caption is written**.

## 2. The workflow

1. Read tasks from a JSON file.
2. Download a video.
3. Use FFmpeg to select a small number of frames from it.
4. Ask a vision-capable model to describe the visible scene in factual language.
5. Ask a text-generation model to produce captions in the requested styles, using the scene description as context.
6. Parse the model output and check that it has the expected structure.
7. Retry failed styles rather than rerunning every caption.
8. Save the results as JSON.

This is a **two-pass pipeline**:
- Pass 1 tries to describe the scene.
- Pass 2 turns that description into different writing styles.

This separation is intended to reduce unsupported details, but it cannot eliminate hallucinations. If the first description is wrong, the later captions can inherit the mistake.

## 3. Key terms

| Term | Simple explanation |
|---|---|
| Multimodal AI | AI that works with more than one kind of input, such as text and images. |
| Vision-language model | A model that can interpret images and work with text. |
| Frame | One still image from a video. |
| Frame sampling | Choosing a limited set of frames instead of analysing every frame. |
| FFmpeg | A command-line tool for processing video and audio. |
| Prompt | Instructions sent to the model. |
| Style conditioning | Telling a model what voice or tone to use. |
| JSON | A structured text format for data, using objects, arrays, keys, and values. |
| Retry | Trying an operation again after a temporary or recoverable failure. |
| Timeout | A limit on how long an operation may take. |
| Concurrency | Working on several tasks during overlapping time periods. |
| Thread pool | A fixed group of worker threads used to process tasks. |
| API key | A secret token that authorises requests to a service. |
| Docker | A tool for packaging an application and its dependencies into a container image. |
| Streamlit | A Python framework for building a simple interactive web interface. |
| Evaluation rubric | A set of criteria used to judge output quality. |

## 4. Where is the code?

| File | Purpose |
|---|---|
| `app.py` | Main batch pipeline: configuration, video handling, model calls, retries, and output writing. |
| `demo_app.py` | Interactive Streamlit dashboard. |
| `evaluate.py` | Calls a judge model to score generated captions for accuracy and tone. |
| `test_input/TASKS.json` | Sample list of videos and requested caption styles. |
| `requirements.txt` | Python dependencies for the batch pipeline. |
| `requirements_demo.txt` | Additional dependencies for the dashboard. |
| `Dockerfile` | Instructions for building a container image. |
| `.github/workflows/ci.yml` | GitHub Actions workflow that checks Python files can be compiled. |
| `.github/workflows/docker.yml` | Docker-related automation. |

## 5. Run the batch pipeline locally

### Requirements

- Python 3.10 or newer is a reasonable starting point; CI uses Python 3.11.
- FFmpeg installed and available on your PATH.
- A Fireworks AI API key.
- Network access to the video URLs and model provider.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/IamAlag/video-caption-agent.git
   cd video-caption-agent
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell instead:
   # .venv\Scripts\Activate.ps1
   ```
3. Install the batch dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Set the API key as an environment variable. On macOS/Linux:
   ```bash
   export FIREWORKS_API_KEY="your_key_here"
   export TASKS_PATH="./test_input/TASKS.json"
   export RESULTS_PATH="./test_output/results.json"
   python app.py
   ```
   In Windows PowerShell, use:
   ```powershell
   $env:FIREWORKS_API_KEY="your_key_here"
   $env:TASKS_PATH="./test_input/TASKS.json"
   $env:RESULTS_PATH="./test_output/results.json"
   python app.py
   ```
5. Inspect the results file. Confirm that each task has captions for the styles requested in the input.

The default paths in the code are container-style paths such as `/input/tasks.json` and `/output/results.json`. Setting `TASKS_PATH` and `RESULTS_PATH` as above makes local runs easier. Ensure the output directory exists if the program expects it to be present.

### Run the demo UI

Install the demo dependencies, then run Streamlit:
```bash
python -m pip install -r requirements_demo.txt
streamlit run demo_app.py
``

### Check Python syntax

```bash
python -m compileall -q app.py demo_app.py evaluate.py
``

This catches syntax/compilation problems. It does not prove the external API calls succeed or that captions are accurate.

## 6. What is in the input JSON?

A task includes:
- `task_id`: an identifier for the video task.
- `video_url`: the video location.
- `styles`: the caption styles to generate.

For example:
```json
{
  "task_id": "demo-1",
  "video_url": "https://example.com/video.mp4",
  "styles": ["formal", "sarcastic"]
}
```

The sample file requests four styles for each listed video. Use the real sample file as the source of truth for the accepted schema.

## 7. What can go wrong?

- **Download fails:** the URL is unavailable, network access fails, or the server rejects the request.
- **No useful frames:** the clip or FFmpeg processing fails; the program may use a fallback sampling strategy.
- **Model timeout or rate limit:** the provider is slow or rejects too many requests.
- **Malformed JSON:** the model returns commentary or Markdown around the data; parsing and retries try to recover.
- **Wrong scene description:** the first pass misreads a frame, and the error can flow into every style.
- **Missed event:** frame sampling only sees selected still images, so an important action between frames may be missed.
- **Poor style:** the caption is factual but does not sound like the requested voice, or vice versa.

## 8. How does evaluation work?

The `evaluate.py` script asks a judge model to score each caption on two dimensions:
- **Accuracy (1–5):** does it match the video?
- **Tone adherence (1–5):** does it match the requested style?

Important: this is an LLM-as-judge evaluation. It is useful as a quick feedback mechanism, but the judge can also be wrong or inconsistent. Do not describe its scores as objective ground truth. Stronger evaluation would include human-reviewed examples, a fixed dataset, and repeated measurements.

## 9. Interview explanation

> "I built a Python video-captioning pipeline that samples frames from each clip, creates a factual scene description, and then generates captions in several requested styles. It includes structured-output parsing, retries for failed styles, concurrent processing, and a separate evaluation script. The main trade-off is that frame sampling reduces processing cost and latency but can miss events between frames. I would improve the project by measuring factual accuracy on a labelled dataset and making failure cases easier to inspect."

## 10. Five-session learning plan

- [ ] **Session 1:** Read this guide and inspect `test_input/TASKS.json`.
- [ ] **Session 2:** Find where `app.py` reads configuration and task input.
- [ ] **Session 3:** Trace one video through frame extraction, pass one, and pass two.
- [ ] **Session 4:** Read the retry and JSON-parsing code; list failure cases.
- [ ] **Session 5:** Run the syntax check and evaluate a small output sample if API access is available.

## Next reading

- [Interview notes](INTERVIEW_NOTES.md) — concise interview answers and trade-offs.
- [Repository README](../README.md) — overview, configuration, and Docker instructions.
