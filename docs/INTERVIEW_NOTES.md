# Interview Notes: Video Captioning Agent

## 30-second explanation

"I built a Python pipeline that takes video URLs, samples representative frames, creates a factual scene description, and then generates captions in multiple writing styles. It includes retries for malformed or failed outputs, concurrent processing, a Docker setup, and a separate evaluation script."

## Two-minute explanation

"The design separates scene understanding from creative writing. First, the pipeline downloads a clip and uses FFmpeg to sample frames. A vision-capable model describes what appears in those frames. A second model uses that description to generate captions in requested styles, such as formal, sarcastic, technical humour, and everyday humour. The program parses the output into a structured format, retries failed styles, and writes results to JSON.

The trade-off is that sampling only a few frames is cheaper and faster than analysing every frame, but it can miss events between frames. The factual first pass can also be wrong, so the second pass may repeat or embellish that error. I would evaluate it using a fixed set of clips, human-reviewed factual labels, style-adherence checks, latency, and cost."

## Key questions

### Why use two passes?
To separate the question "What is visible?" from "How should we phrase it?" This can help constrain creative captions to factual context, but it is not a guarantee against hallucination.

### Why not analyse every frame?
Processing every frame may increase cost, latency, and payload size. Sampling reduces work but can miss short events. Frame count and selection strategy should be chosen based on evaluation.

### Why use JSON?
JSON gives downstream code a predictable structure to validate and save. Model output is not guaranteed to be valid JSON, so defensive parsing and retries are needed.

### Why retry only a failed style?
If three styles are valid and one is malformed, retrying just the failed one avoids repeating successful work and may reduce cost and latency.

### What is concurrency?
Multiple video tasks can make progress at the same time. A thread pool provides a bounded number of workers. Too much concurrency can overload local resources or trigger provider rate limits.

### How is it evaluated?
The evaluation script asks another model to score accuracy and tone adherence from 1 to 5. This is an LLM-as-judge approach, so the scores are estimates rather than ground truth. A stronger setup adds human review and a repeatable labelled dataset.

### What are the limitations?
- Sparse frames can miss actions.
- The scene description can be wrong.
- The creative pass can add unsupported details.
- Network downloads and external model APIs can fail.
- Model judging can be inconsistent.
- Provider cost and latency need to be measured.

## Useful engineering terms

- **Timeout:** stop waiting after a defined period.
- **Retry:** repeat a failed operation, ideally with limits.
- **Fallback:** use an alternate method when the preferred method fails.
- **Concurrency limit:** maximum number of tasks processed at once.
- **Structured output:** data following a defined shape, such as JSON.
- **Observability:** logs and metrics that help explain what happened during a run.

## What I would improve next

1. Add unit tests for JSON parsing and retry behaviour.
2. Save structured error records for failed downloads, frame extraction, and model responses.
3. Add a small human-labelled dataset with expected factual details.
4. Compare frame-sampling strategies on accuracy, runtime, and cost.
5. Track model version, prompts, retries, duration, and token/cost estimates per task.
6. Add safeguards around video URL downloading before accepting arbitrary user-supplied URLs in a public service.

## Be honest in interviews

- Say this is a prototype/hackathon project.
- Do not say two-pass generation eliminates hallucinations.
- Do not present LLM-judge scores as objective truth.
- Do not invent throughput, accuracy, or cost numbers.
- Explain one trade-off and how you would test it.

## Practice checklist

- [ ] Explain the pipeline without reading the README.
- [ ] Explain why frames are sampled.
- [ ] Explain how the two passes differ.
- [ ] Describe a failure that retries can recover from.
- [ ] Name a failure that retries cannot solve, such as a consistently incorrect scene description.
- [ ] Explain how you would measure quality.
