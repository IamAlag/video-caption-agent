"""
Self-evaluation script: Simulates the LLM-Judge scoring for Track 2.
Run this after generating results to check quality before submission.

Usage:
  set FIREWORKS_API_KEY=your_key
  python evaluate.py [results_path]
"""

import os
import sys
import json
import requests
import time

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
# Use Kimi K2 as judge
JUDGE_MODEL = os.environ.get(
    "JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p6"
)

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_output/results.json"

RUBRIC = """You are an expert judge evaluating video captions. Score each caption on two dimensions:

## Scoring Rubric

### Accuracy (1-5)
1 = Completely wrong/hallucinated content
2 = Major inaccuracies about the video content
3 = Partially accurate but missing key details
4 = Mostly accurate with minor issues
5 = Perfectly describes the actual video content

### Tone Adherence (1-5)
1 = Completely wrong tone (e.g., formal caption that's funny)
2 = Tone is vaguely present but poorly executed
3 = Tone is recognizable but inconsistent
4 = Good tone execution with minor slips
5 = Perfect tone - unmistakably matches the style

## Style Definitions
- formal: Professional, objective, no humor, no personality, BBC documentary narrator
- sarcastic: Dry, ironic, British wit, understatement, never mean
- humorous_tech: Programming jokes, dev culture references, bugs/deployments/Git
- humorous_non_tech: Warm everyday humor, family/food/pets, no tech jargon

## Output Format
For each style, respond with ONLY a JSON object:
{"style": "...", "accuracy": N, "tone": N, "feedback": "one sentence explanation"}
"""


def evaluate_caption(task_id: str, style: str, caption: str) -> dict:
    """Evaluate a single caption using the LLM judge."""
    prompt = (
        f"{RUBRIC}\n\n"
        f"## Caption to Evaluate\n"
        f"Task: {task_id}\n"
        f"Style: {style}\n"
        f"Caption: \"{caption}\"\n\n"
        f"Score this caption. Respond with ONLY the JSON object."
    )

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{FIREWORKS_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1,
                "thinking": {"type": "disabled"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        import re
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  [error] Judging {task_id}/{style}: {e}")

    return {"style": style, "accuracy": 0, "tone": 0, "feedback": "evaluation failed"}


def main():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    print(f"Evaluating {len(results)} results from {RESULTS_PATH}")
    print(f"Judge model: {JUDGE_MODEL}")
    print("=" * 60)

    all_scores = []

    for result in results:
        task_id = result["task_id"]
        captions = result["captions"]
        print(f"\n[Video] {task_id}")

        for style, caption in captions.items():
            score = evaluate_caption(task_id, style, caption)
            all_scores.append(score)

            acc = score.get("accuracy", "?")
            tone = score.get("tone", "?")
            feedback = score.get("feedback", "")
            status_char = "OK" if (isinstance(acc, (int, float)) and isinstance(tone, (int, float)) and acc >= 4 and tone >= 4) else "WARN" if (isinstance(acc, (int, float)) and isinstance(tone, (int, float)) and acc >= 3 and tone >= 3) else "FAIL"

            print(f"  [{status_char}] {style:20s} | Accuracy: {acc}/5 | Tone: {tone}/5 | {feedback}")
            time.sleep(0.5)  # rate limiting

    # Summary
    valid = [s for s in all_scores if s.get("accuracy", 0) > 0]
    if valid:
        avg_acc = sum(s["accuracy"] for s in valid) / len(valid)
        avg_tone = sum(s["tone"] for s in valid) / len(valid)
        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"  Average Accuracy: {avg_acc:.1f}/5")
        print(f"  Average Tone:     {avg_tone:.1f}/5")
        print(f"  Combined Score:   {(avg_acc + avg_tone) / 2:.1f}/5")
        print(f"  Captions Evaluated: {len(valid)}")

        # Style breakdown
        print(f"\n  Per-Style Breakdown:")
        for style in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]:
            style_scores = [s for s in valid if s.get("style") == style]
            if style_scores:
                sa = sum(s["accuracy"] for s in style_scores) / len(style_scores)
                st = sum(s["tone"] for s in style_scores) / len(style_scores)
                print(f"    {style:20s}: Accuracy {sa:.1f} | Tone {st:.1f}")

        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
