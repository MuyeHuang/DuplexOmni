#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NExTQA 整库唯一视频事件打标 → 单行 jsonl（可并发、断点续传、debug 试跑）。

输出 schema 见文件末尾「JSONL 字段说明」；方案 A：event 中不包含任何「打断」类字段。

依赖：openai、pandas 或 pyarrow（读 parquet）；ffprobe 可选（时长）。
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 复用同目录单条打标脚本中的解析与路径工具
from auto_tag_nextqa_events import (
    _normalize_event_intervals,
    ffprobe_duration_sec,
    resolve_video_playback_info,
    safe_json_loads,
)

SCHEMA_VERSION = "video_event_tags.v1"
TAGGING_PROMPT_ID = "nextqa_event_batch_v2_detail"

DEFAULT_NEXTQA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "realtime_serving"
    / "huggingface.co"
    / "datasets"
    / "lmms-lab"
    / "NExTQA"
)
# 解压后的视频文件目录（与 parquet 中 video id 拼成可 ffprobe 的路径）
DEFAULT_VIDEO_ROOT = (DEFAULT_NEXTQA_ROOT / "videos").resolve()

# (config_name, split_folder_name, parquet_filename)  —— split 用于 meta 与优先级
PARQUET_MANIFEST: List[Tuple[str, str, str]] = [
    ("OE", "train", "train-00000-of-00001.parquet"),
    ("OE", "validation", "validation-00000-of-00001.parquet"),
    ("OE", "test", "test-00000-of-00001.parquet"),
    ("MC", "test", "test-00000-of-00001.parquet"),
]

PRIORITY_ORDER = {
    ("OE", "train"): 0,
    ("OE", "validation"): 1,
    ("OE", "test"): 2,
    ("MC", "test"): 3,
}

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "video_stream_output"
    / "video_tags"
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"

BATCH_SYSTEM_PROMPT = """You are a video event tagger for NExTQA-style clips.
You will receive the actual video and metadata together.
Infer scene/action boundaries from visual content, using metadata only as auxiliary context.

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

Do NOT include any other keys inside event objects (no interrupt / no assistant control fields).

Rules:
1) Intervals are contiguous: E1.start=0, E(i).end = E(i+1).start, last end = video duration (when duration is known).
2) keyframe_sec == start_sec for every event.
3) When duration_sec is known, all times must lie in [0, duration_sec].
4) Event COUNT must follow duration (roughly):
   - duration < 15s → 1–2 events
   - 15s <= duration < 45s → 2–4 events
   - 45s <= duration < 120s → 3–6 events
   - duration >= 120s → 4–10 events
5) For long clips, do NOT use a single event spanning almost the whole video; split on likely visual/story beats.
6) If duration is unknown, still output a reasonable segmentation and explain in global_notes.
7) Keep event count natural (do NOT create extra segments just for detail); add detail inside each event instead.
8) event_title should include the core action + actor/object when possible (not generic titles like "Action continues").
9) event_summary must be specific and concrete: include actor(s), action, object(s), and scene/context cues visible in the clip.
10) Avoid vague summaries ("something happens", "person does activity"); mention observable evidence.
"""


def canonical_video_id(video: Any) -> str:
    if video is None:
        return ""
    if isinstance(video, bool):
        return str(video).lower()
    if isinstance(video, int):
        return str(video)
    if isinstance(video, float):
        if video != video:
            return ""
        return str(int(video)) if float(int(video)) == video else str(video)
    s = str(video).strip()
    if s.isdigit():
        return s
    return s


def resolve_video_file_strict(video_id: str, video_root: str) -> Path:
    """
    严格定位视频文件：必须存在，否则视为失败（不允许文本 fallback）。
    """
    root = Path(video_root)
    candidates = [
        root / video_id,
        root / f"{video_id}.mp4",
        root / "NExTVideo" / f"{video_id}.mp4",
        root / f"{video_id}.webm",
        root / f"{video_id}.mkv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"video_not_found: {video_id}; checked={', '.join(str(x) for x in candidates)}"
    )


def filter_ids_with_existing_videos(video_ids: List[str], video_root: str) -> List[str]:
    kept: List[str] = []
    for cid in video_ids:
        try:
            _ = resolve_video_file_strict(cid, video_root)
            kept.append(cid)
        except FileNotFoundError:
            continue
    return kept


def load_parquet_records(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    except Exception:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()


def build_video_aggregates(nextqa_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    全局按 canonical_video_id 聚合；代表行取 PRIORITY 最小（OE train 优先）。
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for config, split_name, fname in PARQUET_MANIFEST:
        sub = "OE" if config == "OE" else "MC"
        ap = nextqa_root / sub / fname
        if not ap.exists():
            raise FileNotFoundError(f"Missing parquet: {ap}")
        pr = PRIORITY_ORDER.get((config, split_name), 99)
        rel = f"{sub}/{fname}"
        rows = load_parquet_records(ap)
        for row in rows:
            raw_v = row.get("video")
            cid = canonical_video_id(raw_v)
            if not cid:
                continue
            key = (config, split_name, rel)
            if cid not in groups:
                groups[cid] = {
                    "canonical_id": cid,
                    "representative": dict(row),
                    "rep_priority": pr,
                    "source_counts": {},  # key str -> count
                    "source_order": [],
                }
            g = groups[cid]
            sc = g["source_counts"]
            sk = f"{config}|{split_name}|{rel}"
            sc[sk] = sc.get(sk, 0) + 1
            if sk not in g["source_order"]:
                g["source_order"].append(sk)
            if pr < g["rep_priority"]:
                g["rep_priority"] = pr
                g["representative"] = dict(row)

    # appears_in 结构
    for cid, g in groups.items():
        appears: List[Dict[str, Any]] = []
        for sk, cnt in g["source_counts"].items():
            cfg, spl, prel = sk.split("|", 2)
            appears.append(
                {
                    "config": cfg,
                    "split": spl,
                    "parquet_relpath": prel,
                    "row_count": cnt,
                }
            )
        g["appears_in"] = sorted(appears, key=lambda x: (x["config"], x["split"]))
    return groups


def estimate_duration_sec(
    rep: Dict[str, Any],
    playback: Dict[str, Any],
    assumed_fps: float,
) -> Tuple[Optional[float], str, float]:
    """返回 (duration_sec, duration_source, assumed_fps_effective)"""
    vp = playback.get("video_path") or ""
    d = ffprobe_duration_sec(vp)
    if d is not None and d > 0:
        return d, "ffprobe", assumed_fps
    fc = rep.get("frame_count")
    try:
        fc_i = int(fc) if fc is not None else 0
    except (TypeError, ValueError):
        fc_i = 0
    if fc_i > 0 and assumed_fps > 0:
        return fc_i / assumed_fps, "framecount", assumed_fps
    return None, "unknown", assumed_fps


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


def strip_forbidden_event_keys(events: List[Dict[str, Any]]) -> None:
    forbidden = (
        "assistant_should_be_interruptible_at_next_event",
        "interrupt",
        "assistant_interrupt",
    )
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for k in forbidden:
            ev.pop(k, None)


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
        "event_id",
        "start_sec",
        "end_sec",
        "keyframe_sec",
        "event_title",
        "event_summary",
        "question_trigger_window",
    )
    required_qtw_keys = (
        "pre_start_sec",
        "pre_end_sec",
        "post_start_sec",
        "post_end_sec",
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
                "pre_start_sec": s,
                "pre_end_sec": s,
                "post_start_sec": e,
                "post_end_sec": e,
            },
        )
    return tags


def build_user_prompt(
    rep: Dict[str, Any],
    playback: Dict[str, Any],
    duration_sec: Optional[float],
    duration_source: str,
    appears_in: List[Dict[str, Any]],
    canonical_id: str,
    extra_hint: str = "",
) -> str:
    payload = {
        "canonical_video_id": canonical_id,
        "raw_video_field": rep.get("video"),
        "duration_sec": duration_sec,
        "duration_source": duration_source,
        "frame_count": rep.get("frame_count"),
        "width": rep.get("width"),
        "height": rep.get("height"),
        "appears_in": appears_in,
        "sample_qid": rep.get("qid"),
        "sample_type": rep.get("type"),
        "sample_question": rep.get("question"),
        "sample_answer": rep.get("answer"),
        "video_path_hint": playback.get("video_path"),
        "video_exists": playback.get("video_exists"),
    }
    base = (
        "You are given the actual video in this request. "
        "Segment it into contiguous events for streaming dialogue.\n"
        "Return JSON only per system schema.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    if extra_hint:
        base += f"\n\n{extra_hint}"
    return base


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
                        "chat_template_kwargs": {"enable_thinking": False},
                        "mm_processor_kwargs": {
                            "fps": video_fps,
                            "do_sample_frames": True,
                        },
                    }
                    if use_mm_processor_kwargs
                    else {"chat_template_kwargs": {"enable_thinking": False}}
                ),
            )
            content = response.choices[0].message.content or ""
            last_content = content
            return content
        except Exception as exc:
            last_err = exc
            time.sleep(1.5)
    raise RuntimeError(f"LLM failed: {last_err}; last_preview={last_content[:200]!r}")


def process_one_video(
    *,
    canonical_id: str,
    rep: Dict[str, Any],
    appears_in: List[Dict[str, Any]],
    video_root: str,
    assumed_fps: float,
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
    video_id = canonical_id
    try:
        video_file = resolve_video_file_strict(video_id, video_root)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "event_tags": {"event_intervals": [], "global_notes": ""},
            "error": str(exc),
            "video_file": None,
            "duration_sec": None,
            "duration_source": "missing_video",
        }

    playback = resolve_video_playback_info(rep.get("video"), video_root)
    playback["video_path"] = str(video_file)
    playback["video_exists"] = True
    playback["video_file_uri"] = video_file.as_uri()
    duration_sec, duration_source, fps_used = estimate_duration_sec(rep, playback, assumed_fps)

    user_base = build_user_prompt(
        rep, playback, duration_sec, duration_source, appears_in, canonical_id
    )

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
        user_text = user_base if round_idx == 0 else user_base + "\n\n" + hint
        tags = None
        for _parse_try in range(parse_retries):
            try:
                raw = call_llm_raw(
                    client,
                    model,
                    BATCH_SYSTEM_PROMPT,
                    user_text,
                    video_file,
                    max_tokens,
                    http_retries,
                    video_fps,
                    use_mm_processor_kwargs,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "event_tags": {"event_intervals": [], "global_notes": ""},
                    "error": f"llm_request:{exc}",
                    "video_file": str(video_file),
                    "duration_sec": duration_sec,
                    "duration_source": duration_source,
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
        strip_forbidden_event_keys(tags.get("event_intervals") or [])
        tags, _ = _normalize_event_intervals(tags, duration_sec)
        tags = fill_missing_output_keys(tags)
        strip_forbidden_event_keys(tags.get("event_intervals") or [])
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
        tags.pop("assistant_should_be_interruptible_at_next_event", None)
        return {
            "ok": True,
            "event_tags": tags,
            "error": None,
            "video_file": str(video_file),
            "duration_sec": duration_sec,
            "duration_source": duration_source,
        }

    return {
        "ok": False,
        "event_tags": (tags if tags is not None else {"event_intervals": [], "global_notes": ""}),
        "error": last_fail or "unknown",
        "video_file": str(video_file),
        "duration_sec": duration_sec,
        "duration_source": duration_source,
    }


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
                m = obj.get("meta") or {}
                vid = m.get("canonical_id") or m.get("video_id")
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
    return output_dir / "video_tags_nextqa_full.jsonl"


def parse_args() -> argparse.Namespace:
    epilog = f"""完整路径示例（请按你本机实际存在与否使用）:
  试跑 10 条:
    python batch_tag_nextqa_videos.py --debug --debug-sample-size 10 --seed 42 --video-root {DEFAULT_VIDEO_ROOT}
  全量续跑:
    python batch_tag_nextqa_videos.py --workers 100 --video-root {DEFAULT_VIDEO_ROOT}
"""
    p = argparse.ArgumentParser(
        description="NExTQA full-corpus video event tagging (jsonl).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--nextqa-root",
        type=str,
        default=str(DEFAULT_NEXTQA_ROOT.resolve()),
        help="NExTQA 数据集根目录（含 OE/、MC/）。",
    )
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.resolve()))
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Exact jsonl path; overrides default naming if set.",
    )
    p.add_argument(
        "--video-root",
        type=str,
        default=str(DEFAULT_VIDEO_ROOT),
        help=f"视频文件根目录（必须可定位到真实视频文件；找不到直接失败）。默认: {DEFAULT_VIDEO_ROOT}",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="发送给 vLLM 的视频采样 fps（mm_processor_kwargs.fps）。",
    )
    p.add_argument(
        "--disable-mm-processor-kwargs",
        action="store_true",
        help="不向服务端发送 mm_processor_kwargs（用于排查/绕过部分 vLLM 版本的处理器兼容问题）。",
    )
    p.add_argument("--assumed-fps", type=float, default=30.0)
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--max-retries", type=int, default=3, help="HTTP-level retries for each LLM call."
    )
    p.add_argument(
        "--parse-retries",
        type=int,
        default=4,
        help="Number of full LLM re-generations if JSON parse fails (same round).",
    )
    p.add_argument(
        "--quality-rounds",
        type=int,
        default=3,
        help="Max outer rounds when validation fails (stricter user hint).",
    )
    p.add_argument(
        "--max-event-span-ratio",
        type=float,
        default=0.45,
        help="Reject if one event span/duration exceeds this (when duration>45s).",
    )
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--debug", action="store_true", help="Random sample N videos and exit.")
    p.add_argument("--debug-sample-size", type=int, default=10)
    p.add_argument(
        "--debug-only-existing-videos",
        action="store_true",
        help="Debug 抽样前仅保留本地可定位到真实视频文件的 video_id（推荐开启）。",
    )
    p.add_argument(
        "--only-existing-videos",
        action="store_true",
        help="全量/续跑前仅保留本地可定位到真实视频文件的 video_id。",
    )
    p.add_argument("--seed", type=int, default=20260427)
    p.add_argument("--run-id", type=str, default="", help="Override run id (default: timestamp).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai") from exc
    nextqa_root = Path(args.nextqa_root)
    out_dir = Path(args.output_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else default_output_path(out_dir, args.debug, run_id)

    print("Loading parquet and aggregating by video_id ...")
    groups = build_video_aggregates(nextqa_root)
    all_ids = sorted(groups.keys())
    print(f"Unique videos: {len(all_ids)}")

    rng = random.Random(args.seed)
    global_pool = all_ids
    if args.only_existing_videos:
        global_pool = filter_ids_with_existing_videos(all_ids, args.video_root)
        print(
            f"Global pool filtered by local files: {len(global_pool)}/{len(all_ids)} "
            "videos exist under --video-root."
        )

    if args.debug:
        debug_pool = global_pool
        if args.debug_only_existing_videos:
            debug_pool = filter_ids_with_existing_videos(global_pool, args.video_root)
            print(
                f"Debug pool filtered by local files: {len(debug_pool)}/{len(global_pool)} "
                "videos exist under --video-root."
            )
        if not debug_pool:
            print("No debug candidates after filtering existing videos.")
            return
        n = min(args.debug_sample_size, len(debug_pool))
        work_ids = sorted(rng.sample(debug_pool, k=n))
        run_mode = "debug"
    else:
        done = read_done_ids(out_path)
        work_ids = [x for x in global_pool if x not in done]
        run_mode = "full"
        print(f"Resume: {len(done)} already in {out_path}, {len(work_ids)} remaining.")

    if not work_ids:
        print("Nothing to do.")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    write_lock = threading.Lock()

    def work(cid: str) -> Tuple[str, Dict[str, Any]]:
        g = groups[cid]
        rep = g["representative"]
        appears = g["appears_in"]
        rep_priority = g["rep_priority"]

        t0 = time.time()
        result = process_one_video(
            canonical_id=cid,
            rep=rep,
            appears_in=appears,
            video_root=args.video_root,
            assumed_fps=args.assumed_fps,
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

        video_file = result.get("video_file")
        if video_file:
            pb = {
                "raw_video_id": str(rep.get("video")),
                "video_path": video_file,
                "video_file_uri": Path(video_file).as_uri(),
                "video_path_candidates": [video_file],
                "video_exists": True,
            }
            d_sec = result.get("duration_sec")
            d_src = result.get("duration_source", "ffprobe")
            fps_u = args.assumed_fps
        else:
            pb = resolve_video_playback_info(rep.get("video"), args.video_root)
            d_sec, d_src, fps_u = estimate_duration_sec(rep, pb, args.assumed_fps)
        # meta 里主 split：appears 中 priority 最小者
        best_pri = min(
            PRIORITY_ORDER.get((x["config"], x["split"]), 99) for x in appears
        )
        main_cfg, main_split = "unknown", "unknown"
        for x in appears:
            if PRIORITY_ORDER.get((x["config"], x["split"]), 99) == best_pri:
                main_cfg, main_split = x["config"], x["split"]
                break

        line = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok" if result["ok"] else "failed",
            "error": result.get("error"),
            "event_tags": result["event_tags"],
            "meta": {
                "source_dataset": "lmms-lab/NExTQA",
                "nextqa_root": str(nextqa_root.resolve()),
                "canonical_id": cid,
                "video_id": cid,
                "video": rep.get("video"),
                "config": main_cfg,
                "split": main_split,
                "rep_priority": rep_priority,
                "appears_in": appears,
                "source_dataset_paths": [str(nextqa_root / x["parquet_relpath"]) for x in appears],
                "qid_sample": rep.get("qid"),
                "type_sample": rep.get("type"),
                "parquet_row_total_across_appears": sum(x["row_count"] for x in appears),
                "frame_count": rep.get("frame_count"),
                "width": rep.get("width"),
                "height": rep.get("height"),
                "video_root": args.video_root,
                "video_path": pb.get("video_path"),
                "video_file_uri": pb.get("video_file_uri", ""),
                "video_path_candidates": pb.get("video_path_candidates", []),
                "video_exists": pb.get("video_exists", False),
                "duration_sec": d_sec,
                "duration_source": d_src,
                "assumed_fps": fps_u,
                "llm": {
                    "base_url": args.base_url,
                    "model": args.model,
                },
                "tagging_prompt_id": TAGGING_PROMPT_ID,
                "run_id": run_id,
                "run_mode": run_mode,
                "debug": args.debug,
                "created_at": datetime.now().isoformat(),
                "processing_seconds": round(elapsed, 3),
            },
        }
        if result["ok"] and line["event_tags"].get("event_intervals") is not None:
            qf: List[str] = []
            if d_sec and d_sec > 60 and len(line["event_tags"]["event_intervals"]) < 2:
                qf.append("suspicious_few_events")
            if qf:
                line["event_tags"]["quality_flags"] = qf
        return cid, line

    print(f"Writing to {out_path} mode={run_mode} workers={args.workers} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, cid): cid for cid in work_ids}
        done_n = 0
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                _, line = fut.result()
                append_line(out_path, line, write_lock)
                done_n += 1
                st = line.get("status")
                print(f"[{done_n}/{len(work_ids)}] {cid} {st}")
            except Exception as exc:
                err_line = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error": f"worker:{exc}",
                    "event_tags": {"event_intervals": [], "global_notes": ""},
                    "meta": {
                        "canonical_id": cid,
                        "run_id": run_id,
                        "run_mode": run_mode,
                        "debug": args.debug,
                    },
                }
                append_line(out_path, err_line, write_lock)
                print(f"[err] {cid} {exc}")

    print(f"Done. Wrote/updated: {out_path}")


# --- JSONL 字段说明（与计划 schema 对齐，方案 A 无打断字段）---
# schema_version: 格式版本
# status / error: 是否成功与失败原因
# event_tags.event_intervals[]: start_sec, end_sec, keyframe_sec(=start), title/summary, question_trigger_window
# event_tags.global_notes: 模型说明
# event_tags.quality_flags: 后处理轻量提示（可选）
# meta: 溯源、时长来源、是否找到视频文件、run_mode(debug|full)、续跑时同输出文件去重 key 为 meta.canonical_id

if __name__ == "__main__":
    main()
