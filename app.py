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
# Model configuration: Kimi K2 for vision, DeepSeek V4 Pro for text/styling
VISION_MODEL_ID = os.environ.get(
    "FIREWORKS_VISION_MODEL", "accounts/fireworks/models/kimi-k2p6"
)
TEXT_MODEL_ID = os.environ.get(
    "FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/deepseek-v4-pro"
)

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "7"))
MAX_FRAME_DIM = int(os.environ.get("MAX_FRAME_DIM", "768"))
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # contest hard limit: 30s per request
MAX_WORKERS = 3       # parallel video processing
TWO_PASS = os.environ.get("TWO_PASS", "1") == "1"  # set to 0 to disable

# --- Style Definitions -------------------------------------------------------

STYLE_PERSONAS = {
    "formal": {
        "voice": (
            "You are a professional BBC World Service documentary narrator. "
            "Write with absolute clinical objectivity."
        ),
        "do": [
            "Use precise, neutral language",
            "Describe only what is visually confirmed in the frames",
            "Maintain third-person perspective throughout",
            "Use present tense for ongoing actions",
        ],
        "dont": [
            "NEVER use humor, irony, metaphors, or personality",
            "NEVER express opinions or subjective judgments",
            "NEVER use exclamation marks or casual language",
            "NEVER speculate about what might be happening off-screen",
        ],
        "examples": [
            "A golden retriever retrieves a tennis ball from a shallow stream while its owner watches from the bank.",
            "Two autonomous delivery vehicles navigate a residential road bordered by mature oak trees.",
            "A chameleon remains stationary on a branch, its skin displaying mottled green and brown patterns.",
            "A runner sprints across the track under stadium lighting during a track and field competition.",
            "A chef slices raw vegetables on a wooden cutting board in a professional restaurant kitchen.",
        ],
    },
    "sarcastic": {
        "voice": (
            "You are a jaded film critic at a festival who has seen everything "
            "and finds the mundane absurd. Use dry British wit and understatement."
        ),
        "do": [
            "Use ironic understatement and deadpan observations",
            "Point out the gap between effort and outcome",
            "Be witty, never mean-spirited",
            "Ground your sarcasm in what is actually visible",
        ],
        "dont": [
            "NEVER be wholesome, sincere, or encouraging",
            "NEVER use tech jargon or programming references",
            "NEVER be cruel or personally attacking subjects",
            "NEVER use obvious sarcasm markers like '/s' or 'NOT'",
            "NEVER explain the joke or comment on the sarcasm itself",
            "NEVER sound like a stand-up comedian -- this is dry, cynical film-criticism, not observational stand-up",
        ],
        "examples": [
            "Oh look, another influencer pretending a brick wall is the Sistine Chapel. Truly groundbreaking composition.",
            "Ah yes, nature's original hide-and-seek champion, blending in with all the effort of someone who just doesn't care.",
            "Riveting footage of a person sitting at a desk. Someone call the Academy.",
            "A chef tosses vegetables with the theatrical flair of someone who knows the camera is rolling. Culinary innovation, 2026.",
            "A runner sprints for first place as if the finish line holds the meaning of life, only to receive a plastic ribbon.",
        ],
    },
    "humorous_tech": {
        "voice": (
            "You are a veteran software engineer who sees the entire world as code. "
            "Everything is a bug, a feature, or a deployment gone wrong."
        ),
        "do": [
            "Use specific programming concepts: recursion, race conditions, null pointers, Git conflicts, CI/CD, stack overflow",
            "The tech metaphor MUST connect to what is actually happening in the video",
            "Make developers laugh with recognition humor",
            "Use current tech culture references (stand-ups, code reviews, 'it works on my machine')",
        ],
        "dont": [
            "NEVER use generic humor without a tech angle",
            "NEVER use outdated or irrelevant tech references",
            "NEVER be formal or dry -- you're the funny person at the stand-up",
            "NEVER just name-drop tech words without a real joke",
        ],
        "examples": [
            "This cat on a keyboard is basically every junior dev pushing to production on a Friday -- chaotic, unplanned, and someone will cry about it on Monday.",
            "These self-driving cars are stuck in an infinite loop with no exit condition, also known as my last sprint.",
            "This chameleon's camouflage algorithm has better backward compatibility than half the APIs I've integrated this year.",
            "This kitchen prep setup is running multiple parallel threads, but the main chef process has blocked the stack with a massive memory leak.",
            "The athlete's start-line response time has lower latency than a simple ping request to our database servers.",
        ],
    },
    "humorous_non_tech": {
        "voice": (
            "You are a warm, observational stand-up comedian. Think about "
            "everyday life, family situations, and universal human experiences."
        ),
        "do": [
            "Use relatable everyday humor -- family, food, pets, commuting, aging",
            "Compare what you see to universal human experiences",
            "Be warm and inclusive -- something everyone would laugh at",
            "Use vivid, specific comparisons (not vague 'funny' statements)",
        ],
        "dont": [
            "NEVER use programming, technology, or internet culture references",
            "NEVER be mean-spirited or exclusionary",
            "NEVER use technical jargon of any kind",
            "NEVER be dry or sarcastic -- you're genuinely amused and warm",
        ],
        "examples": [
            "That dog fetching the ball looks exactly like me sprinting to the microwave when it beeps.",
            "My grandmother trying to parallel park has more confidence than these little robot cars on a Sunday stroll.",
            "This lizard sitting perfectly still is giving 'me pretending to be asleep when someone asks for help moving furniture' energy.",
            "Watching this chef toss pancakes is like watching my father try to flip a mattress -- highly energetic and bound to end in disaster.",
            "This sprinter running in the rain looks like my family running from the car to the house when we realize it's pouring.",
        ],
    },
}

FALLBACK_CAPTIONS = {
    "formal": "The video depicts a sequence of scenes involving various subjects and activities in a structured environment.",
    "sarcastic": "Oh wonderful, another video of things happening in a place. Truly the content the world was crying out for.",
    "humorous_tech": "This video has more frames than my browser tabs after a Stack Overflow rabbit hole, and about as much resolution on the actual problem.",
    "humorous_non_tech": "Watching this is like waiting for your food at a restaurant -- you know something is coming, you are just not sure what or when.",
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
    "You are a precise visual analyst. Examine these frames from a short video clip.\n\n"
    "Describe EXACTLY what you see. Be specific about:\n"
    "1. Main subject(s) — species, breed, color, clothing, posture, expression\n"
    "2. Actions — exact movement, speed, direction, interaction with objects\n"
    "3. Setting — indoor/outdoor, time of day, weather, architecture, vegetation\n"
    "4. Notable details — specific colors, textures, brands, numbers, spatial relationships\n\n"
    "Write 3-5 dense, specific sentences. Use concrete nouns and active verbs. "
    "Avoid vague words like 'something,' 'someone,' 'various,' 'scenes.' "
    "Do NOT guess or transcribe text on signs, buildings, or screens unless it is clearly and sharply legible. "
    "Do NOT speculate about audio or unseen events."
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
        "model": VISION_MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 400,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
    }

    result = api_call(payload)
    if result and "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    return ""


# --- Pass 2: Styled Captioning -----------------------------------------------

def build_style_prompt(styles: list, scene_description: str = "") -> str:
    """Build a detailed style prompt, optionally grounded in a scene description."""
    style_blocks = []
    for s in styles:
        persona = STYLE_PERSONAS.get(s, {})
        voice = persona.get("voice", "")
        dos = "\n".join(f"    + {d}" for d in persona.get("do", []))
        donts = "\n".join(f"    - {d}" for d in persona.get("dont", []))
        examples = "\n".join(f'    e.g. "{ex}"' for ex in persona.get("examples", []))

        style_blocks.append(
            f'### Style: "{s}"\n'
            f"  Voice: {voice}\n"
            f"  DO:\n{dos}\n"
            f"  DON'T:\n{donts}\n"
            f"  Examples (for reference, do NOT copy):\n{examples}"
        )

    style_text = "\n\n".join(style_blocks)
    keys_example = ", ".join(f'"{s}": "Your caption here"' for s in styles)

    scene_block = ""
    if scene_description:
        scene_block = (
            f"## Scene Description (verified facts about this video)\n"
            f"{scene_description}\n\n"
        )

    return (
        "You are an expert video captioner with perfect tone control.\n\n"
        f"{scene_block}"
        "## Your Task\n"
        "Based on the visual frames"
        + (" and the scene description above" if scene_description else "")
        + ", write one caption per style.\n\n"
        "## Critical Rules (Mandatory for Accuracy and Tone):\n"
        "1. VISUAL GROUNDING: Every caption MUST describe ONLY what is visible in the frames. Do NOT speculate or add details that cannot be visually confirmed.\n"
        "2. NO HALLUCINATIONS: Do NOT invent off-screen characters, names, backstories, or scenarios. If a person appears, describe them by appearance, NOT by invented roles or names.\n"
        "3. COMPARISON RULE: For humorous styles, use explicit similes ('looks like', 'resembles') so comparisons are clearly figurative. The actual visual subject MUST remain the grammatical subject.\n"
        "4. TONE SEPARATION: Each style voice must be unmistakable (formal=clinical/objective; sarcastic=dry/ironic; humorous_tech=dev-culture jokes; humorous_non_tech=warm everyday humor).\n"
        "5. LENGTH: Each caption must be EXACTLY 1 sentence. No more. Brevity is part of the style.\n"
        "6. DETAILED AND VIVID: Keep captions highly descriptive and detailed (e.g. mention colors, clothes, specific species/objects shown in the Scene Description) rather than using generic words like 'person', 'car', or 'animal'.\n"
        "7. ANTI-BLENDING: The four captions must be COMPLETELY DIFFERENT in voice, vocabulary, and sentence structure. If two captions could have been written by the same person, you have FAILED. formal=clinical encyclopedia entry, sarcastic=dry film critic at 2am, humorous_tech=frustrated dev doing stand-up, humorous_non_tech=warm comedian at a family dinner.\n\n"
        f"## Style Definitions\n{style_text}\n\n"
        "## Output Format\n"
        "Respond with ONLY a valid JSON object. No markdown fences, no commentary.\n"
        f"Exact shape: {{{keys_example}}}"
    )


def pass2_styled_captions(frame_paths: list, styles: list, scene_description: str = "") -> dict:
    """
    Pass 2: Generate styled captions.
    Uses DeepSeek V4 Pro (text-only) if scene description is available;
    falls back to Kimi K2 (vision) if running single-pass.
    """
    prompt = build_style_prompt(styles, scene_description)

    if scene_description:
        # Use fast, powerful text-only model
        payload = {
            "model": TEXT_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.40,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
    else:
        # Fallback to vision model if scene description is missing
        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(fp)}"},
            })
        payload = {
            "model": VISION_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 600,
            "temperature": 0.30,
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
    persona = STYLE_PERSONAS.get(style, {})
    voice = persona.get("voice", "")
    dos = "; ".join(persona.get("do", []))
    examples = " | ".join(f'"{ex}"' for ex in persona.get("examples", [])[:2])

    scene_block = f"\nScene: {scene_description}\n" if scene_description else ""

    prompt = (
        f"Write exactly ONE caption in the '{style}' style for this video.\n"
        f"{scene_block}\n"
        f"Voice: {voice}\n"
        f"DO: {dos}\n"
        f"Examples: {examples}\n\n"
        "Write 1-2 sentences. Respond with ONLY the caption text, nothing else."
    )

    if scene_description:
        payload = {
            "model": TEXT_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.40,
            "thinking": {"type": "disabled"},
        }
    else:
        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths[:3]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(fp)}"},
            })
        payload = {
            "model": VISION_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 200,
            "temperature": 0.30,
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
    styles = task.get("styles", list(STYLE_PERSONAS.keys()))

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

        # Use configured frame count (default: 7)
        duration = get_duration(video_path)
        num_frames = NUM_FRAMES

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
    print(f"Vision model: {VISION_MODEL_ID}, Text model: {TEXT_MODEL_ID}")
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
                        styles = task.get("styles", list(STYLE_PERSONAS.keys()))
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