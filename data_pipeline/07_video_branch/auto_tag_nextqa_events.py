#!/usr/bin/env python3
import argparse
import ast
import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_SEED = 20260427
DEFAULT_OUTPUT_DIR = (
    "outputs/video_stream/video_tags"
)
DEFAULT_INPUT_CANDIDATES = [
    "datasets/lmms-lab/NExTQA/OE/train-00000-of-00001.parquet",
    "datasets/lmms-lab/NExTQA/OE/test-00000-of-00001.parquet",
]


SYSTEM_PROMPT = """You are a video event tagger.
Given one sample's metadata, build event-interval annotations for streaming dialogue training.

You MUST output valid JSON only, with this schema (no extra keys in each event):
{
  "event_intervals": [
    {
      "event_id": "E1",
      "start_sec": 0.0,
      "end_sec": 3.0,
      "keyframe_sec": 0.0,
      "event_title": "short phrase",
      "event_summary": "one concise sentence",
      "question_trigger_window": {
        "pre_start_sec": 0.0,
        "pre_end_sec": 0.0,
        "post_start_sec": 0.0,
        "post_end_sec": 0.0
      }
    }
  ],
  "global_notes": "short note"
}

Rules:
1) Event intervals are continuous and non-overlapping.
2) keyframe_sec should be the beginning of each event.
3) Do NOT force event count. Use the natural number of events in the video.
4) Ensure times are ascending and in seconds.
5) If video duration is known, all times must be within [0, duration].
6) Event intervals should be continuous in time and should not leave unexplained gaps.
7) Do not include interrupt or assistant control fields.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample NExTQA items and generate event tags with local vLLM."
    )
    parser.add_argument("--input-file", help="Path to parquet/jsonl file. Optional.")
    parser.add_argument("--video-root", default="", help="Video root directory. Optional.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def choose_input_file(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    for candidate in DEFAULT_INPUT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError(
        "No default input file found. Please pass --input-file explicitly."
    )


def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        # 优先 pandas；失败后回退 pyarrow，避免环境依赖差异导致不可用。
        try:
            import pandas as pd

            data = pd.read_parquet(path)
            return data.to_dict(orient="records")
        except Exception:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Read parquet failed. Please install at least one of: pandas or pyarrow."
                ) from exc
            table = pq.read_table(path)
            return table.to_pylist()

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported input format: {path.suffix}")


def resolve_video_path(video_value: Any, video_root: str) -> str:
    if video_value is None:
        return ""
    video_str = str(video_value).strip()
    if not video_str:
        return ""
    if video_root:
        root = Path(video_root)
        candidate = root / video_str
        return str(candidate.resolve())
    return video_str


def resolve_video_playback_info(video_value: Any, video_root: str) -> Dict[str, Any]:
    """
    生成可播放视频路径信息：
    - raw_video_id: 数据集原始 video 字段
    - video_path: 首选解析路径（优先 video_root）
    - file_uri: 若路径存在则给出 file:// URI
    """
    raw = "" if video_value is None else str(video_value).strip()
    preferred = resolve_video_path(video_value, video_root)
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    if raw and raw not in candidates:
        candidates.append(raw)

    # 尝试常见后缀，给用户更易播放的路径
    extended: List[str] = []
    for c in candidates:
        p = Path(c)
        if p.suffix:
            extended.append(str(p))
            continue
        extended.append(str(p))
        extended.append(str(p.with_suffix(".mp4")))
        extended.append(str(p.with_suffix(".webm")))
        extended.append(str(p.with_suffix(".mkv")))
    dedup = []
    seen = set()
    for item in extended:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)

    existing = [x for x in dedup if Path(x).exists()]
    best = existing[0] if existing else (dedup[0] if dedup else "")
    file_uri = Path(best).resolve().as_uri() if best and Path(best).exists() else ""
    return {
        "raw_video_id": raw,
        "video_path": best,
        "video_file_uri": file_uri,
        "video_path_candidates": dedup[:10],
        "video_exists": bool(best and Path(best).exists()),
    }


def ffprobe_duration_sec(video_path: str) -> Optional[float]:
    if not video_path:
        return None
    path = Path(video_path)
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            return None
        return float(output)
    except Exception:
        return None


def build_user_prompt(sample: Dict[str, Any], idx: int, duration_sec: Optional[float]) -> str:
    payload = {
        "sample_index": idx,
        "qid": sample.get("qid"),
        "video": sample.get("video_path"),
        "duration_sec": duration_sec,
        "frame_count": sample.get("frame_count"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "type": sample.get("type"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "additional_ref_answer": sample.get("additional_ref_answer"),
    }
    return (
        "Create streaming dialogue-oriented event interval tags for this sample.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


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
    # 移除 } 或 ] 前的多余逗号
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

    # 兼容部分模型吐出 python dict 风格内容
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


def call_model(
    client: Any,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
    last_error = None
    last_content = ""
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content or ""
        last_content = content
        try:
            return safe_json_loads(content)
        except Exception as exc:
            last_error = exc
            # 继续重试
    raise RuntimeError(
        f"Model response parse failed after {max_retries} attempts: {last_error}; "
        f"raw_content_preview={last_content[:300]!r}"
    )


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


def sample_records(records: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if not records:
        raise ValueError("Input dataset is empty.")
    k = min(sample_size, len(records))
    rng = random.Random(seed)
    return rng.sample(records, k=k)


def write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it in this env first, e.g. `pip install openai`."
        ) from exc

    input_path = choose_input_file(args.input_file)
    records = load_records(input_path)
    picked = sample_records(records, args.sample_size, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)

    manifest: List[Dict[str, Any]] = []
    for i, item in enumerate(picked, start=1):
        playback = resolve_video_playback_info(item.get("video"), args.video_root)
        video_path = playback["video_path"]
        duration_sec = ffprobe_duration_sec(video_path)

        sample = dict(item)
        sample["video_path"] = video_path
        prompt = build_user_prompt(sample, i, duration_sec)
        tags = call_model(
            client=client,
            model=args.model,
            user_prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
        tags, normalized_fix_count = _normalize_event_intervals(tags, duration_sec)

        qid = str(item.get("qid", f"idx_{i}"))
        out_name = f"{run_id}_sample{i:02d}_qid_{qid}.json"
        out_path = output_dir / out_name

        result_payload = {
            "meta": {
                "run_id": run_id,
                "input_file": str(input_path),
                "sample_index": i,
                "qid": item.get("qid"),
                "video": item.get("video"),
                "raw_video_id": playback["raw_video_id"],
                "video_path": video_path,
                "video_file_uri": playback["video_file_uri"],
                "video_path_candidates": playback["video_path_candidates"],
                "video_exists": playback["video_exists"],
                "duration_sec": duration_sec,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "type": item.get("type"),
                "normalized_fix_count": normalized_fix_count,
            },
            "event_tags": tags,
        }
        write_result(out_path, result_payload)
        manifest.append({"sample_index": i, "qid": item.get("qid"), "file": str(out_path)})
        print(f"[OK] wrote {out_path}")

    manifest_path = output_dir / f"{run_id}_manifest.json"
    write_result(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "input_file": str(input_path),
            "sample_size": len(picked),
            "items": manifest,
        },
    )
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
