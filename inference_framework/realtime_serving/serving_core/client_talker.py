"""
Client for server_talker: read req_*.pt chunks from disk, POST in sorted order, concatenate WAV output.

No vllm-omni. Uses requests + stdlib wave.

Thinker ``HiddenDiskStore`` payload (after training-alignment fix) may include top-level
``generated_token_ids`` (length n-1, aligned rows) and optional ``completion_token_ids_full``.
This client does not interpret those fields; it forwards the file bytes unchanged. The HTTP
service reads only ``turns[0].top_hidden_state`` and ``turns[0].text_embedding`` (same row count).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import uuid
from pathlib import Path

import requests


def _sort_key_req_pt(p: Path) -> tuple[int, str]:
    m = re.match(r"req_(\d+)\.pt$", p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (10**18, p.name)


def silence_wav_bytes(*, sample_rate: int = 24000, num_frames: int = 480) -> bytes:
    """一段静音 WAV（默认 24kHz、约 20ms），用于某 chunk 无音频时仍占住时间轴以便拼接。"""
    import wave

    pcm = (b"\x00\x00") * num_frames
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wo:
        wo.setnchannels(1)
        wo.setsampwidth(2)
        wo.setframerate(sample_rate)
        wo.writeframes(pcm)
    return buf.getvalue()


def concat_wav_bytes(parts: list[bytes]) -> bytes:
    import wave

    if not parts:
        return b""
    rates: list[int] = []
    frames: list[bytes] = []
    for c in parts:
        with wave.open(io.BytesIO(c), "rb") as w:
            rates.append(w.getframerate())
            frames.append(w.readframes(w.getnframes()))
    if len(set(rates)) != 1:
        raise RuntimeError(f"Sample rate mismatch across chunks: {rates}")
    rate = rates[0]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wo:
        wo.setnchannels(1)
        wo.setsampwidth(2)
        wo.setframerate(rate)
        for f in frames:
            wo.writeframes(f)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description="Send thinker chunk .pt files to server_talker and save concatenated WAV.")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:20000",
        help="server_talker base URL (no trailing slash)",
    )
    p.add_argument(
        "--chunks-dir",
        type=Path,
        required=True,
        help="Directory containing req_*.pt (e.g. thinker_hidden_store/chunks)",
    )
    p.add_argument(
        "--output-wav",
        type=Path,
        required=True,
        help="Output concatenated WAV path",
    )
    p.add_argument(
        "--session-id",
        default="",
        help="Optional session id; default UUID for one run",
    )
    args = p.parse_args()
    session_id = args.session_id.strip() or str(uuid.uuid4())

    root = args.chunks_dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    paths = sorted([p for p in root.glob("req_*.pt")], key=_sort_key_req_pt)
    if not paths:
        print(f"No req_*.pt under {root}", file=sys.stderr)
        sys.exit(1)

    # 一次性读盘，避免按 chunk 交替读盘与请求；瓶颈留在服务端推理与网络。
    chunks: list[tuple[Path, bytes]] = [(p, p.read_bytes()) for p in paths]

    url = args.base_url.rstrip("/") + "/v1/talker/chunk"
    wav_parts: list[bytes] = []
    for i, (pt_path, blob) in enumerate(chunks):
        data = {"session_id": session_id}
        files = {"file": (pt_path.name, io.BytesIO(blob), "application/octet-stream")}
        r = requests.post(url, data=data, files=files, timeout=3600)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} for {pt_path.name}: {r.text[:500]}", file=sys.stderr)
            sys.exit(1)
        ct = r.headers.get("content-type", "")
        meta_hdr = r.headers.get("X-Talker-Meta", "")
        if meta_hdr:
            print(f"Chunk {pt_path.name} X-Talker-Meta: {meta_hdr}", file=sys.stderr)
        if "application/json" in ct:
            print(f"Chunk {pt_path.name} JSON body: {r.text[:500]}", file=sys.stderr)
            wav_parts.append(silence_wav_bytes())
            print(
                f"  -> padded silence (no wav), {i + 1}/{len(chunks)}",
                flush=True,
                file=sys.stderr,
            )
            continue
        if "audio" not in ct and "octet-stream" not in ct and "wav" not in ct:
            pass
        wav_parts.append(r.content)
        print(f"OK {i + 1}/{len(chunks)} {pt_path.name} wav_bytes={len(r.content)}", flush=True)

    out = concat_wav_bytes(wav_parts)
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    args.output_wav.write_bytes(out)
    print(f"Wrote {args.output_wav} ({len(out)} bytes) session_id={session_id}")


if __name__ == "__main__":
    main()
