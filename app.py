"""
AMD Developer Hackathon: ACT II -- Track 2 (Video Captioning Agent)

Reads /input/tasks.json, for each task:
  1. Downloads the video clip
  2. Samples frames with ffmpeg (scene-change detection + uniform fallback)
  3. Pass 1: Gets a factual scene description from Kimi K2
  4. Pass 2: Generates all 4 styled captions grounded in the scene description
  5. Writes /output/results.json

Model: Kimi K2 (kimi-k2p6) via Fireworks AI API
IMPORTANT: thinking must be disabled to stay under 30s per request.
"""

import os
import json
import base64
import subprocess
import tempfile
import time
import re
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Track execution time for the circuit breaker
START_TIME = time.time()

# --- Configuration -----------------------------------------------------------

TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")

# Read API key from environment (loaded dynamically to support late setting/imports)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
# Kimi K2 -- fast vision model with thinking disabled (~7s per clip)
MODEL_ID = os.environ.get(
    "FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6"
)

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "7"))
MAX_FRAME_DIM = int(os.environ.get("MAX_FRAME_DIM", "768"))
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # contest hard limit: 30s per request
MAX_WORKERS = 3       # parallel video processing
TWO_PASS = os.environ.get("TWO_PASS", "1") == "1"  # set to 0 to disable

# --- Style Definitions -------------------------------------------------------

STYLE_VOICES = {
    "formal": "Professional documentary narrator. Clinical, objective, third-person, present tense. Zero humor, zero personality, zero opinions.",
    "sarcastic": "Jaded film critic. Dry British wit, ironic understatement, deadpan. Never wholesome, never mean, no tech jargon.",
    "humorous_tech": "Software engineer doing stand-up. Everything is a bug, deployment, or race condition. Must use specific programming concepts.",
    "humorous_non_tech": "Warm observational comedian. Family, food, pets, everyday life comparisons. Never technical, never sarcastic.",
}

FALLBACK_CAPTIONS = {
    "formal": "The video depicts various subjects and activities in a structured environment.",
    "sarcastic": "Another video of things happening in a place, truly riveting cinema.",
    "humorous_tech": "This video has more unresolved frames than my bug tracker.",
    "humorous_non_tech": "Watching this is like waiting for your food at a restaurant.",
}


# --- Video Processing --------------------------------------------------------

def download_video(url: str, out_path: str) -> None:
    """Download video using yt-dlp with a requests streaming fallback."""
    # Try yt-dlp first (handles YouTube, Shorts, Reels, Vimeo, TikTok, etc.)
    try:
        import yt_dlp
        ydl_opts = {
            'outtmpl': out_path,
            'format': 'mp4/best',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return
    except Exception as e:
        print(f"[warn] yt-dlp download failed, falling back to direct stream: {e}")

    # Fallback to direct requests streaming (handles direct S3/GCS MP4 links)
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            return
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Failed to download video from {url}: {last_err}")


def get_duration(video_path: str) -> float:
    """Get video duration via ffprobe with timeout protection."""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=15
        )
        return float(probe.stdout.strip())
    except Exception:
        return 30.0


def extract_frames_scene_detect(video_path: str, out_dir: str, max_frames: int = 5) -> list:
    """
    Extract frames at scene changes using ffmpeg's scene detection filter.
    Falls back to uniform sampling if scene detection yields too few/many frames.
    """
    scene_out = os.path.join(out_dir, "scene_%03d.jpg")
    scene_filter = "select='gt(scene,0.25)'" + f",scale={MAX_FRAME_DIM}:{MAX_FRAME_DIM}:force_original_aspect_ratio=decrease"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", scene_filter,
                "-frames:v", str(max_frames + 4),
                "-vsync", "vfr",
                "-q:v", "2",
                scene_out,
            ],
            capture_output=True, text=True, timeout=20
        )
    except subprocess.TimeoutExpired:
        print(f"[warn] scene detection timed out for {video_path}")

    scene_frames = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("scene_") and f.endswith(".jpg")
        and os.path.getsize(os.path.join(out_dir, f)) > 0
    ])

    if len(scene_frames) >= 3:
        if len(scene_frames) > max_frames:
            # Sample max_frames evenly across the detected scenes
            step = len(scene_frames) / max_frames
            sampled = [scene_frames[int(i * step)] for i in range(max_frames)]
            # Clean up the unsampled ones
            for f in scene_frames:
                if f not in sampled:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            return sampled
        return scene_frames

    # Clean up scene frames before fallback
    for f in scene_frames:
        try:
            os.remove(f)
        except OSError:
            pass

    return extract_frames_uniform(video_path, out_dir, max_frames)


def extract_frames_uniform(video_path: str, out_dir: str, num_frames: int = 5) -> list:
    """Uniform frame extraction with resize in a single pass to prevent timeouts."""
    duration = get_duration(video_path)
    fps_val = num_frames / max(1.0, duration)
    out_pattern = os.path.join(out_dir, "frame_%03d.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps=fps={fps_val:.4f},scale={MAX_FRAME_DIM}:{MAX_FRAME_DIM}:force_original_aspect_ratio=decrease",
                "-q:v", "2", out_pattern,
            ],
            capture_output=True, timeout=20
        )
    except subprocess.TimeoutExpired:
        print(f"[warn] uniform frame extraction timed out for {video_path}")

    frame_paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
        and os.path.getsize(os.path.join(out_dir, f)) > 0
    ])

    # If rounding or framing issues resulted in fewer frames, duplicate the last frame
    if 0 < len(frame_paths) < num_frames:
        last_frame = frame_paths[-1]
        while len(frame_paths) < num_frames:
            frame_paths.append(last_frame)

    return frame_paths[:num_frames]


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- API Calls ---------------------------------------------------------------

def api_call(payload: dict, max_retries: int = MAX_RETRIES) -> dict | None:
    """
    Make Fireworks API call with exponential backoff.
    Returns parsed response dict or None on failure.
    """
    api_key = FIREWORKS_API_KEY or os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        print("  [error] FIREWORKS_API_KEY is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{FIREWORKS_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt + random.uniform(0, 1), 15)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} -- waiting {wait:.1f}s")
            time.sleep(wait)

    print(f"  [error] API call failed after {max_retries} attempts: {last_err}")
    return None


# --- Pass 1: Scene Understanding ---------------------------------------------

SCENE_PROMPT = (
    "Describe exactly what is visible in these video frames in 2-4 specific sentences.\n\n"
    "Include: main subject (appearance, color, species/type), actions, setting, notable objects.\n"
    "Use concrete nouns and active verbs. No vague words.\n"
    "Do NOT guess text on signs, buildings, or screens unless clearly legible.\n"
    "Do NOT speculate about audio, off-screen events, or identities."
)


def pass1_scene_understanding(frame_paths: list) -> str:
    """
    Pass 1: Get a factual description of the video content.
    This grounds the style generation in verified visual facts.
    """
    content = [{"type": "text", "text": SCENE_PROMPT}]
    for fp in frame_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(fp)}"},
        })

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
    }

    result = api_call(payload)
    if result and "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    return ""


# --- Pass 2: Styled Captioning -----------------------------------------------

def build_style_prompt(styles: list, scene_description: str = "") -> str:
    """Build a compact style prompt (~500 tokens instead of ~2000)."""
    style_lines = "\n".join(f"- {s}: {STYLE_VOICES[s]}" for s in styles)
    keys_example = ", ".join(f'"{s}": "caption"' for s in styles)

    scene_block = ""
    if scene_description:
        scene_block = f"Scene: {scene_description}\n\n"

    return (
        f"{scene_block}"
        "Write one short caption per style for this video.\n\n"
        f"Styles:\n{style_lines}\n\n"
        "Rules:\n"
        "- Maximum 25 words per caption. Shorter is better.\n"
        "- Describe ONLY what is visible. No guessed text on signs/screens. No invented characters, names, or backstories.\n"
        "- Each caption must sound like a COMPLETELY different person wrote it.\n"
        "- For humor styles, use similes ('looks like', 'resembles') to keep comparisons clearly figurative.\n\n"
        f"Respond with ONLY valid JSON: {{{keys_example}}}"
    )


def pass2_styled_captions(frame_paths: list, styles: list, scene_description: str = "") -> dict:
    """
    Pass 2: Generate styled captions, optionally grounded in scene description.
    """
    prompt = build_style_prompt(styles, scene_description)

    content = [{"type": "text", "text": prompt}]
    for fp in frame_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(fp)}"},
        })

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 400,
        "temperature": 0.50,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }

    result = api_call(payload)
    if result and "choices" in result:
        text = result["choices"][0]["message"]["content"].strip()
        return parse_json_response(text, styles)
    return {s: "" for s in styles}


# --- Per-Style Retry ----------------------------------------------------------

def retry_single_style(frame_paths: list, style: str, scene_description: str = "") -> str:
    """Retry a single failed style with a focused prompt."""
    voice = STYLE_VOICES.get(style, "")
    scene_block = f"Scene: {scene_description}\n" if scene_description else ""

    prompt = (
        f"Write ONE caption (max 25 words) in the '{style}' style for this video.\n"
        f"{scene_block}"
        f"Voice: {voice}\n"
        "Only describe what is visible. Respond with ONLY the caption text."
    )

    content = [{"type": "text", "text": prompt}]
    for fp in frame_paths[:3]:  # fewer frames for retry
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(fp)}"},
        })

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "temperature": 0.50,
        "thinking": {"type": "disabled"},
    }

    result = api_call(payload, max_retries=2)
    if result and "choices" in result:
        text = result["choices"][0]["message"]["content"].strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    return ""


# --- JSON Parsing -------------------------------------------------------------

def parse_json_response(text: str, styles: list) -> dict:
    """
    Robust JSON extraction from model response.
    Handles: case-insensitivity, markdown fences, stray text, partial JSON.
    """
    text = text.strip()

    # Remove markdown code fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.split("\n")[0].strip().lower() in ("json", ""):
                inner = "\n".join(inner.split("\n")[1:])
            text = inner.strip()

    def normalize_key(k: str) -> str:
        return str(k).lower().strip().replace("_", "").replace("-", "").replace(" ", "")

    def extract_from_dict(d: dict) -> dict:
        normalized = {normalize_key(k): v for k, v in d.items()}
        res = {}
        for s in styles:
            norm_s = normalize_key(s)
            res[s] = str(normalized.get(norm_s, "")).strip()
        return res

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return extract_from_dict(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return extract_from_dict(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: regex key-value extraction
    result = {}
    for s in styles:
        # Match case-insensitively and allow optional spaces or underscores
        norm_s = normalize_key(s)
        # Regex to match key with any casing/spacing
        pattern = rf'"[^"]*{norm_s}[^"]*"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[s] = match.group(1).strip()

    if any(result.get(s) for s in styles):
        return {s: result.get(s, "") for s in styles}

    return {s: "" for s in styles}


# --- Task Processing ---------------------------------------------------------

def process_task(task: dict, tmp_base: str) -> dict:
    """Process a single video captioning task."""
    task_id = task.get("task_id", f"task_{int(time.time() * 1000) % 100000}")
    video_url = task.get("video_url", "")
    styles = task.get("styles", list(STYLE_VOICES.keys()))

    # Per-task temp directory to avoid frame collisions in parallel mode
    tmp_dir = os.path.join(tmp_base, task_id)
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[{task_id}] Starting...")

    try:
        if not video_url:
            raise ValueError("Missing video_url in task specification")

        # Download video
        video_path = os.path.join(tmp_dir, f"{task_id}.mp4")
        print(f"[{task_id}] Downloading video...")
        download_video(video_url, video_path)

        # Get duration and calculate adaptive frame count (1 frame per 15s, capped at 5 to 8 frames)
        duration = get_duration(video_path)
        if "NUM_FRAMES" in os.environ:
            num_frames = int(os.environ["NUM_FRAMES"])
        else:
            num_frames = min(8, max(5, int(duration / 15)))

        # Extract frames (scene detection with uniform fallback)
        print(f"[{task_id}] Extracting {num_frames} frames (duration: {duration:.1f}s)...")
        frames = extract_frames_scene_detect(video_path, tmp_dir, num_frames)
        if not frames:
            frames = extract_frames_uniform(video_path, tmp_dir, num_frames)
        if not frames:
            raise RuntimeError("No frames could be extracted from video")
        print(f"[{task_id}] Got {len(frames)} frames")

        # Circuit breaker: if overall time elapsed > 360s, disable two-pass to guarantee no TIMEOUT
        use_two_pass = TWO_PASS
        if time.time() - START_TIME > 360:
            print(f"[{task_id}] [warn] Time elapsed > 360s, forcing fast single-pass mode to prevent timeout")
            use_two_pass = False

        scene_description = ""
        if use_two_pass:
            # Pass 1: Scene Understanding
            print(f"[{task_id}] Pass 1: Scene understanding...")
            scene_description = pass1_scene_understanding(frames)
            if scene_description:
                print(f"[{task_id}] Scene: {scene_description[:80]}...")
            else:
                print(f"[{task_id}] Pass 1 returned empty, proceeding without grounding")

        # Pass 2 (or single pass): Styled Captioning
        print(f"[{task_id}] {'Pass 2' if use_two_pass else 'Generating'}: Styled captions...")
        captions = pass2_styled_captions(frames, styles, scene_description)

        # Retry any missing styles individually
        missing = [s for s in styles if not captions.get(s)]
        if missing:
            print(f"[{task_id}] Retrying {len(missing)} missing styles: {missing}")
            for s in missing:
                retried = retry_single_style(frames, s, scene_description)
                if retried:
                    captions[s] = retried
                    print(f"[{task_id}] Recovered '{s}'")

    except Exception as e:
        print(f"[error] Task {task_id} failed: {e}")
        captions = {s: "" for s in styles}

    # Final fallback: ensure no empty captions
    for s in styles:
        if not captions.get(s):
            captions[s] = FALLBACK_CAPTIONS.get(s, "A video showing various scenes and activities.")
            print(f"[{task_id}] Used fallback for '{s}'")

    print(f"[{task_id}] Done")
    return {"task_id": task_id, "captions": captions}


# --- Main --------------------------------------------------------------------

def main() -> None:
    # Ensure output directory exists
    if RESULTS_PATH:
        dir_name = os.path.dirname(RESULTS_PATH)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    tasks = []
    try:
        if not os.path.exists(TASKS_PATH):
            print(f"[warn] Tasks file not found at {TASKS_PATH}. Writing empty results file.")
            with open(RESULTS_PATH, "w") as f:
                json.dump([], f)
            return

        with open(TASKS_PATH, "r") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[error] Failed to load tasks from {TASKS_PATH}: {e}. Writing empty results file.")
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump([], f)
        except Exception as write_err:
            print(f"[CRITICAL] Could not even write empty results fallback: {write_err}")
        return

    print(f"Loaded {len(tasks)} tasks from {TASKS_PATH}")
    print(f"Model: {MODEL_ID}")
    print(f"Frames: {NUM_FRAMES}, Two-pass: {TWO_PASS}")

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        if len(tasks) <= 2:
            # Sequential for small batches
            for task in tasks:
                results.append(process_task(task, tmp_dir))
        else:
            # Parallel for larger batches
            workers = min(MAX_WORKERS, len(tasks))
            print(f"Processing {len(tasks)} videos with {workers} workers...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(process_task, task, tmp_dir): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        task_id = task.get("task_id", "unknown_task")
                        print(f"[error] Task {task_id} exception: {e}")
                        styles = task.get("styles", list(STYLE_VOICES.keys()))
                        results.append({
                            "task_id": task_id,
                            "captions": {
                                s: FALLBACK_CAPTIONS.get(s, "A video showing various scenes.")
                                for s in styles
                            },
                        })

    # Sort by task_id for consistent ordering
    results.sort(key=lambda r: r.get("task_id", ""))

    try:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Done. Wrote {len(results)} results to {RESULTS_PATH}")
        for r in results:
            print(f"  {r.get('task_id')}: {list(r.get('captions', {}).keys())}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"[CRITICAL] Failed to write results to {RESULTS_PATH}: {e}")
        # Try local fallback
        fallback_path = "./results.json"
        try:
            with open(fallback_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[info] Successfully wrote fallback results to {fallback_path}")
        except Exception as ex:
            print(f"[CRITICAL] local fallback write failed as well: {ex}")
        raise e


if __name__ == "__main__":
    main()