#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaVA-Video-178K 视频事件打标 → 单行 JSONL（纯视频输入，无元数据依赖）。
thinking 已开启，max_tokens 默认 8192。

用法:
  python batch_tag_llava_videos.py --debug --debug-sample-size 10 --seed 42
  python batch_tag_llava_videos.py --workers 100
  python batch_tag_llava_videos.py --source-dirs ActivityNet-QA NextQA
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "video_event_tags_llava.v1"
TAGGING_PROMPT_ID = "llava_event_batch_v1"

LLAVA_ROOT = Path(__file__).resolve().parents[2] / "huggingface.co" / "datasets" / "lmms-lab" / "LLaVA-Video-178K"
DEFAULT_SOURCE_DIRS = ["ActivityNet-QA", "NextQA", "liwei_youtube_videos"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "video_stream_output" / "video_tags_llava"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"

# ============================================================
# system prompt
# ============================================================

BATCH_SYSTEM_PROMPT = """You are a video event tagger for streaming dialogue training.
You will receive a video.  Watch it and segment it into contiguous events.

You MUST output a single JSON object only, with exactly these top-level keys:
{
  "event_intervals": [ ... ],
  "global_notes": "short note"
}

Each element of event_intervals MUST have:
- "event_id": "E1", "E2", ...
- "start_sec", "end_sec" (seconds, floats)
- "keyframe_sec": MUST equal "start_sec" for that event (segment start)
- "event_title": short phrase
- "event_summary": one detailed English sentence grounded in visible evidence
- "question_trigger_window": {
    "pre_start_sec", "pre_end_sec", "post_start_sec", "post_end_sec"
  } (all numbers; can be equal if no pre/post window)

Do NOT include any other keys inside event objects.

Rules:
1) Intervals are contiguous: E1.start=0, E(i).end = E(i+1).start, last end = video duration (when duration is known).
2) keyframe_sec == start_sec for every event.
3) When duration_sec is known, all times must lie in [0, duration_sec].
4) Event COUNT must follow duration (roughly):
   - duration < 15s → 1–2 events
   - 15s ≤ duration < 45s → 2–4 events
   - 45s ≤ duration < 120s → 3–6 events
   - duration ≥ 120s → 4–10 events
5) For long clips, do NOT use a single event spanning almost the whole video; split on likely visual/story beats.
6) If duration is unknown, still output a reasonable segmentation and explain in global_notes.
7) Keep event count natural; add detail inside each event instead.
8) event_title should include the core action + actor/object when possible.
9) event_summary must be specific and concrete: include actor(s), action, object(s), and scene/context cues visible in the clip.
10) Avoid vague summaries ("something happens", "person does activity"); mention observable evidence.
"""


# ============================================================
# JSON / text parsing utilities
# ============================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _clean_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _THINK_RE.sub("", t).strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    return t.strip()


def _extract_json_candidate(text: str) -> str:
    t = _clean_text(text)
    if not t:
        return ""
    fence = _FENCE_RE.search(t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()
    return t


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _ensure_dict_with_intervals(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Parsed object is not dict.")
    if "event_intervals" not in obj:
        raise ValueError("Model response missing event_intervals.")
    if not isinstance(obj["event_intervals"], list):
        raise ValueError("event_intervals is not list.")
    return obj


def safe_json_loads(text: str) -> Dict[str, Any]:
    candidate = _extract_json_candidate(text)
    if not candidate:
        raise ValueError("Empty model response.")

    attempts: List[str] = [
        candidate,
        _remove_trailing_commas(candidate),
    ]

    py_like = candidate
    py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    attempts.append(py_like)
    attempts.append(_remove_trailing_commas(py_like))

    errors: List[str] = []
    for idx, raw in enumerate(attempts, start=1):
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
            return _ensure_dict_with_intervals(parsed)
        except Exception as exc_json:
            errors.append(f"json#{idx}: {exc_json}")
        try:
            parsed_py = ast.literal_eval(raw)
            return _ensure_dict_with_intervals(parsed_py)
        except Exception as exc_ast:
            errors.append(f"ast#{idx}: {exc_ast}")

    raise ValueError("Failed to parse model JSON. " + " | ".join(errors[:6]))


# ============================================================
# video utilities
# ============================================================

def ffprobe_duration_sec(video_path: str) -> Optional[float]:
    if not video_path:
        return None
    p = Path(video_path)
    if not p.exists():
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(p),
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not output:
            return None
        return float(output)
    except Exception:
        return None


def collect_videos(
    llava_root: Path,
    source_dirs: List[str],
) -> List[Tuple[str, Path]]:
    """返回 [(video_id, absolute_path), ...]，按 video_id 排序。"""
    results: List[Tuple[str, Path]] = []
    for sd in source_dirs:
        src = llava_root / sd
        if not src.is_dir():
            print(f"[warn] source dir not found, skip: {src}")
            continue
        for vf in sorted(src.rglob("*")):
            if not vf.is_file():
                continue
            if vf.suffix.lower() not in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
                continue
            rel = vf.relative_to(llava_root)
            video_id = str(rel)
            results.append((video_id, vf.resolve()))
    results.sort(key=lambda x: x[0])
    return results


def collect_videos_from_jsonl(input_jsonl: Path) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    seen = set()
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta") or {}
            video_id = meta.get("video_id")
            video_file = meta.get("video_file")
            if not video_id or not video_file or video_id in seen:
                continue
            seen.add(video_id)
            results.append((str(video_id), Path(video_file)))
    return results


# ============================================================
# event tag normalization & validation
# ============================================================

def _normalize_event_intervals(
    tags: Dict[str, Any], duration_sec: Optional[float]
) -> Tuple[Dict[str, Any], int]:
    intervals = tags.get("event_intervals") or []
    fixed = 0
    norm: List[Dict[str, Any]] = []
    for i, event in enumerate(intervals, start=1):
        if not isinstance(event, dict):
            fixed += 1
            continue
        e = dict(event)
        e["event_id"] = str(e.get("event_id") or f"E{i}")
        for key in ("start_sec", "end_sec", "keyframe_sec"):
            try:
                e[key] = float(e.get(key, 0.0))
            except Exception:
                e[key] = 0.0
                fixed += 1
        if e["end_sec"] < e["start_sec"]:
            e["end_sec"] = e["start_sec"]
            fixed += 1
        if duration_sec is not None:
            e["start_sec"] = max(0.0, min(e["start_sec"], duration_sec))
            e["end_sec"] = max(0.0, min(e["end_sec"], duration_sec))
            e["keyframe_sec"] = max(0.0, min(e["keyframe_sec"], duration_sec))
        norm.append(e)
    norm.sort(key=lambda x: x.get("start_sec", 0.0))
    tags["event_intervals"] = norm
    return tags, fixed


def expected_event_bounds(duration_sec: Optional[float]) -> Tuple[int, int]:
    if duration_sec is None or duration_sec <= 0:
        return 1, 8
    d = duration_sec
    if d < 15:
        return 1, 2
    if d < 45:
        return 2, 4
    if d < 120:
        return 3, 6
    return 4, 10


def snap_and_check_continuity(
    tags: Dict[str, Any],
    duration_sec: Optional[float],
    eps: float = 0.08,
) -> Tuple[bool, str]:
    intervals = tags.get("event_intervals") or []
    if not intervals:
        return False, "empty_event_intervals"
    for i, ev in enumerate(intervals, start=1):
        ev["event_id"] = ev.get("event_id") or f"E{i}"
        try:
            ev["keyframe_sec"] = float(ev.get("keyframe_sec", ev.get("start_sec", 0.0)))
            ev["start_sec"] = float(ev.get("start_sec", 0.0))
            ev["end_sec"] = float(ev.get("end_sec", 0.0))
        except (TypeError, ValueError):
            return False, "non_numeric_time"
        ev["keyframe_sec"] = ev["start_sec"]

    intervals.sort(key=lambda x: x["start_sec"])
    if duration_sec is not None and duration_sec > 0:
        intervals[0]["start_sec"] = 0.0
        intervals[0]["keyframe_sec"] = 0.0
        intervals[-1]["end_sec"] = duration_sec
    for i in range(len(intervals) - 1):
        if abs(intervals[i]["end_sec"] - intervals[i + 1]["start_sec"]) > eps:
            return False, "gaps_or_overlap"
    for ev in intervals:
        if ev["end_sec"] < ev["start_sec"]:
            return False, "end_before_start"
    return True, ""


def validate_event_structure(
    tags: Dict[str, Any],
    duration_sec: Optional[float],
    max_event_span_ratio: float,
) -> Tuple[bool, str]:
    intervals = tags.get("event_intervals") or []
    n = len(intervals)
    if n == 0:
        return False, "no_events"
    mn, mx = expected_event_bounds(duration_sec)
    if n < mn or n > mx:
        return False, f"event_count_out_of_range want_{mn}_{mx} got_{n}"

    if duration_sec and duration_sec > 60 and n < 2:
        return False, "too_few_events_long_video"

    if duration_sec and duration_sec > 45:
        for ev in intervals:
            span = float(ev.get("end_sec", 0)) - float(ev.get("start_sec", 0))
            if span / duration_sec > max_event_span_ratio:
                return False, "single_span_too_large"
    return True, ""


def validate_required_output_keys(tags: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(tags, dict):
        return False, "tags_not_object"
    if "event_intervals" not in tags:
        return False, "missing_top_key:event_intervals"
    if "global_notes" not in tags:
        return False, "missing_top_key:global_notes"
    if not isinstance(tags.get("event_intervals"), list):
        return False, "event_intervals_not_list"
    if not isinstance(tags.get("global_notes"), str):
        return False, "global_notes_not_string"

    required_event_keys = (
        "event_id", "start_sec", "end_sec", "keyframe_sec",
        "event_title", "event_summary", "question_trigger_window",
    )
    required_qtw_keys = (
        "pre_start_sec", "pre_end_sec", "post_start_sec", "post_end_sec",
    )
    for i, ev in enumerate(tags.get("event_intervals") or []):
        if not isinstance(ev, dict):
            return False, f"event_not_object:{i}"
        for k in required_event_keys:
            if k not in ev:
                return False, f"missing_event_key:{i}:{k}"
        if not isinstance(ev.get("event_title"), str) or not ev.get("event_title", "").strip():
            return False, f"invalid_event_title:{i}"
        if not isinstance(ev.get("event_summary"), str) or not ev.get("event_summary", "").strip():
            return False, f"invalid_event_summary:{i}"
        qtw = ev.get("question_trigger_window")
        if not isinstance(qtw, dict):
            return False, f"qtw_not_object:{i}"
        for qk in required_qtw_keys:
            if qk not in qtw:
                return False, f"missing_qtw_key:{i}:{qk}"
            try:
                _ = float(qtw[qk])
            except (TypeError, ValueError):
                return False, f"qtw_non_numeric:{i}:{qk}"
    return True, ""


def fill_missing_output_keys(tags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tags, dict):
        return {"event_intervals": [], "global_notes": ""}
    tags.setdefault("event_intervals", [])
    tags.setdefault("global_notes", "")
    if not isinstance(tags["event_intervals"], list):
        tags["event_intervals"] = []
    if not isinstance(tags["global_notes"], str):
        tags["global_notes"] = str(tags["global_notes"])
    for ev in tags["event_intervals"]:
        if not isinstance(ev, dict):
            continue
        s = ev.get("start_sec", 0.0)
        e = ev.get("end_sec", s)
        ev.setdefault(
            "question_trigger_window",
            {
                "pre_start_sec": s, "pre_end_sec": s,
                "post_start_sec": e, "post_end_sec": e,
            },
        )
    return tags


# ============================================================
# LLM call  (thinking ENABLED)
# ============================================================

def call_llm_raw(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    video_file: Path,
    max_tokens: int,
    http_retries: int,
    video_fps: float,
    use_mm_processor_kwargs: bool,
) -> str:
    last_err = None
    last_content = ""
    for _ in range(http_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": video_file.as_uri()},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                extra_body=(
                    {
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                    if use_mm_processor_kwargs
                    else {
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                ),
            )
            content = response.choices[0].message.content or ""
            last_content = content
            return content
        except Exception as exc:
            last_err = exc
            time.sleep(1.5)
    raise RuntimeError(f"LLM failed: {last_err}; last_preview={last_content[:200]!r}")


# ============================================================
# process one video
# ============================================================

def process_one_video(
    *,
    video_id: str,
    video_file: Path,
    client: Any,
    model: str,
    max_tokens: int,
    http_retries: int,
    parse_retries: int,
    quality_rounds: int,
    max_event_span_ratio: float,
    video_fps: float,
    use_mm_processor_kwargs: bool,
) -> Dict[str, Any]:
    if not video_file.exists():
        return {
            "ok": False,
            "event_tags": {"event_intervals": [], "global_notes": ""},
            "error": f"video_not_found: {video_file}",
            "duration_sec": None,
        }

    duration_sec = ffprobe_duration_sec(str(video_file))
    duration_source = "ffprobe" if duration_sec is not None else "unknown"

    user_prompt = (
        "Watch this video and segment it into contiguous events for streaming dialogue training.\n"
        "Return JSON only per system schema.\n"
    )
    if duration_sec is not None:
        user_prompt += f"\nVideo duration: {duration_sec:.1f}s"

    tags: Optional[Dict[str, Any]] = None
    last_fail = ""

    for round_idx in range(quality_rounds):
        hint = ""
        if round_idx > 0:
            hint = (
                "STRICT RETRY: Previous output failed validation ("
                f"{last_fail}). "
                "Respect min/max event counts for this duration; "
                "ensure contiguous intervals covering [0,duration]; "
                "split long single-span events."
            )
        user_text = user_prompt if round_idx == 0 else user_prompt + "\n\n" + hint

        for _parse_try in range(parse_retries):
            try:
                raw = call_llm_raw(
                    client, model, BATCH_SYSTEM_PROMPT, user_text,
                    video_file, max_tokens, http_retries,
                    video_fps, use_mm_processor_kwargs,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "event_tags": {"event_intervals": [], "global_notes": ""},
                    "error": f"llm_request:{exc}",
                    "duration_sec": duration_sec,
                }
            try:
                tags = safe_json_loads(raw)
                tags = fill_missing_output_keys(tags)
                break
            except Exception as exc:
                last_fail = f"json_parse:{exc}"

        if tags is None:
            last_fail = last_fail or "json_parse_exhausted"
            continue

        tags, _ = _normalize_event_intervals(tags, duration_sec)
        tags = fill_missing_output_keys(tags)

        ok_keys, key_reason = validate_required_output_keys(tags)
        if not ok_keys:
            last_fail = key_reason
            continue
        ok_snap, snap_reason = snap_and_check_continuity(tags, duration_sec)
        if not ok_snap:
            last_fail = snap_reason
            continue
        ok_val, val_reason = validate_event_structure(tags, duration_sec, max_event_span_ratio)
        if not ok_val:
            last_fail = val_reason
            continue
        return {
            "ok": True,
            "event_tags": tags,
            "error": None,
            "duration_sec": duration_sec,
            "duration_source": duration_source,
        }

    return {
        "ok": False,
        "event_tags": (tags if tags is not None else {"event_intervals": [], "global_notes": ""}),
        "error": last_fail or "unknown",
        "duration_sec": duration_sec,
        "duration_source": duration_source,
    }


# ============================================================
# checkpoint / output
# ============================================================

def read_done_ids(output_path: Path) -> set:
    done: set = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") != "ok":
                    continue
                vid = (obj.get("meta") or {}).get("video_id")
                if vid:
                    done.add(str(vid))
            except json.JSONDecodeError:
                continue
    return done


def append_line(path: Path, line_obj: Dict[str, Any], lock: threading.Lock) -> None:
    s = json.dumps(line_obj, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(s)
            f.flush()


def default_output_path(output_dir: Path, debug: bool, run_id: str) -> Path:
    if debug:
        return output_dir / f"debug_{run_id}.jsonl"
    return output_dir / "video_tags_llava_full.jsonl"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaVA-Video-178K video event tagging (pure video, no metadata).",
    )
    p.add_argument(
        "--llava-root",
        type=str,
        default=str(LLAVA_ROOT.resolve()),
        help="LLaVA-Video-178K 根目录。",
    )
    p.add_argument(
        "--source-dirs",
        nargs="*",
        default=DEFAULT_SOURCE_DIRS,
        help="要扫描的子目录名。默认为已解压的三个目录。",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.resolve()),
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="精确输出 JSONL 路径；设此参数时忽略默认命名。",
    )
    p.add_argument(
        "--input-jsonl",
        type=str,
        default="",
        help="从已有 JSONL 的 meta.video_id/meta.video_file 读取任务，避免扫描目录。",
    )
    p.add_argument("--video-fps", type=float, default=2.0,
                   help="发送给 vLLM 的视频采样 fps。")
    p.add_argument("--disable-mm-processor-kwargs", action="store_true",
                   help="不发送 mm_processor_kwargs。")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="单次请求最大输出 token（含 thinking）。")
    p.add_argument("--max-retries", type=int, default=3,
                   help="HTTP 级重试次数。")
    p.add_argument("--parse-retries", type=int, default=4,
                   help="同一轮内 JSON 解析失败时的 LLM 重生成次数。")
    p.add_argument("--quality-rounds", type=int, default=3,
                   help="验证失败时最多重试的外层轮数。")
    p.add_argument("--max-event-span-ratio", type=float, default=0.45,
                   help="单个事件 span/duration 超过此值则拒绝（duration>45s 时）。")
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--debug", action="store_true",
                   help="随机采样 N 个视频后退出。")
    p.add_argument("--debug-sample-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--run-id", type=str, default="",
                   help="覆盖 run_id（默认时间戳）。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai") from exc

    llava_root = Path(args.llava_root)
    out_dir = Path(args.output_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else default_output_path(out_dir, args.debug, run_id)

    # --- collect all videos ---
    if args.input_jsonl:
        print(f"Loading video list from {args.input_jsonl} ...")
        all_pairs = collect_videos_from_jsonl(Path(args.input_jsonl))
    else:
        print(f"Scanning {args.source_dirs} under {llava_root} ...")
        all_pairs = collect_videos(llava_root, args.source_dirs)
    print(f"Found {len(all_pairs)} videos total.")
    if not all_pairs:
        print("No videos found. Exiting.")
        return

    # --- decide work set ---
    rng = random.Random(args.seed)
    if args.debug:
        n = min(args.debug_sample_size, len(all_pairs))
        work_pairs = sorted(rng.sample(all_pairs, k=n))
        run_mode = "debug"
    else:
        done = read_done_ids(out_path)
        work_pairs = [(vid, vp) for vid, vp in all_pairs if vid not in done]
        run_mode = "full"
        print(f"Resume: {len(done)} already in {out_path}, {len(work_pairs)} remaining.")

    if not work_pairs:
        print("Nothing to do.")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    write_lock = threading.Lock()

    def work(video_id: str, video_file: Path) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        result = process_one_video(
            video_id=video_id,
            video_file=video_file,
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            http_retries=args.max_retries,
            parse_retries=args.parse_retries,
            quality_rounds=args.quality_rounds,
            max_event_span_ratio=args.max_event_span_ratio,
            video_fps=args.video_fps,
            use_mm_processor_kwargs=not args.disable_mm_processor_kwargs,
        )
        elapsed = time.time() - t0

        source_dir = video_id.split("/")[0] if "/" in video_id else "unknown"

        line = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok" if result["ok"] else "failed",
            "error": result.get("error"),
            "event_tags": result["event_tags"],
            "meta": {
                "video_id": video_id,
                "video_file": str(video_file),
                "video_file_uri": video_file.as_uri(),
                "source_dir": source_dir,
                "llava_root": str(llava_root.resolve()),
                "duration_sec": result.get("duration_sec"),
                "duration_source": result.get("duration_source", "unknown"),
                "llm": {
                    "base_url": args.base_url,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "thinking_enabled": True,
                },
                "tagging_prompt_id": TAGGING_PROMPT_ID,
                "run_id": run_id,
                "run_mode": run_mode,
                "debug": args.debug,
                "created_at": datetime.now().isoformat(),
                "processing_seconds": round(elapsed, 3),
            },
        }
        return video_id, line

    print(f"Writing to {out_path}  mode={run_mode}  workers={args.workers} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, vid, vf): vid for vid, vf in work_pairs}
        done_n = 0
        for fut in as_completed(futs):
            vid = futs[fut]
            try:
                _, line = fut.result()
                append_line(out_path, line, write_lock)
                done_n += 1
                st = line.get("status")
                print(f"[{done_n}/{len(work_pairs)}] {vid}  {st}")
            except Exception as exc:
                err_line = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error": f"worker:{exc}",
                    "event_tags": {"event_intervals": [], "global_notes": ""},
                    "meta": {
                        "video_id": vid,
                        "run_id": run_id,
                        "run_mode": run_mode,
                        "debug": args.debug,
                    },
                }
                append_line(out_path, err_line, write_lock)
                print(f"[err] {vid}  {exc}")

    print(f"Done.  Wrote/updated: {out_path}")


if __name__ == "__main__":
    main()
