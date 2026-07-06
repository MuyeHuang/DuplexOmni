import asyncio
import base64
import io
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import warnings
import wave
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Sequence

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoProcessor

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_ROOT = os.environ.get(
    "VLLM_ROOT",
    os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "vllm_qwen3_omni")),
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if VLLM_ROOT not in sys.path:
    sys.path.insert(0, VLLM_ROOT)

# Make the module importable by spawned vLLM workers when this file is run
# directly via `python server_thinker.py`.
sys.modules.setdefault("server_thinker", sys.modules[__name__])
os.environ.setdefault("VLLM_USE_V1", "1")

# ``collective_rpc`` returns dicts whose tensor fields are Msgpack-encoded. By default
# ``envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD`` is 256: tensors larger than that use an **aux buffer
# index** as triple[2] (see ``vllm/v1/serial_utils.py`` ``MsgpackEncoder._encode_tensor``). The
# EngineCore ``MsgpackDecoder`` resolves indices using ``aux_buffers`` from the ZMQ multipart
# message; after partial decode to plain Python lists, that index **cannot** be resolved in
# ``server_thinker`` (``data_kind: int``). Raising the threshold forces **inline** buffers
# (``msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, ...)``) for all realistically-sized tensors so
# ``_tensor_from_vllm_msgpack_triple`` can rebuild ``torch.Tensor`` on the API process.
# Must be set **before** importing vLLM so encoder/env modules read the intended value.
_MSGPACK_INLINE_MIN = 2**40  # 1 TiB: any single tensor in this service stays below this
_raw_mt = os.environ.get("VLLM_MSGPACK_ZERO_COPY_THRESHOLD")
if _raw_mt is None:
    os.environ["VLLM_MSGPACK_ZERO_COPY_THRESHOLD"] = str(_MSGPACK_INLINE_MIN)
else:
    try:
        if int(_raw_mt) < 2**20:
            # e.g. default 256 from shell would break API-side triple decode for ~100KiB+ tensors
            warnings.warn(
                f"VLLM_MSGPACK_ZERO_COPY_THRESHOLD={_raw_mt} is too small for inline "
                f"collective_rpc tensors; setting {_MSGPACK_INLINE_MIN}",
                RuntimeWarning,
                stacklevel=2,
            )
            os.environ["VLLM_MSGPACK_ZERO_COPY_THRESHOLD"] = str(_MSGPACK_INLINE_MIN)
    except ValueError:
        pass

from vllm import SamplingParams  # noqa: E402
from vllm import envs as vllm_envs  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker  # noqa: E402
from vllm.v1.worker.gpu_model_runner import (  # noqa: E402
    AsyncGPUModelRunnerOutput,
    GPUModelRunner as VllmGPUModelRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("server_thinker")
LOGGER.info(
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD=%s (see vllm/v1/serial_utils.py; large value avoids "
    "aux-buffer indices in collective_rpc tensor triples on the API process)",
    vllm_envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD,
)


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.environ.get("THINKER_HOST", "0.0.0.0")
    port: int = int(os.environ.get("THINKER_PORT", "19999"))
    tensor_parallel_size: int = int(os.environ.get("THINKER_TP", "4"))
    max_model_len: int = int(os.environ.get("THINKER_MAX_MODEL_LEN", "32386"))
    model: str = os.environ.get(
        "THINKER_MODEL",
        "models/qwen3-omni-thinker",
    )
    served_model_name: str = os.environ.get(
        "THINKER_MODEL",
        "models/qwen3-omni-thinker",
    )
    enable_prefix_caching: bool = True
    limit_mm_per_prompt: dict[str, int] = None  # type: ignore[assignment]
    gpu_memory_utilization: float = float(os.environ.get("THINKER_GPU_MEM_UTIL", "0.9"))
    dtype: str = os.environ.get("THINKER_DTYPE", "auto")
    max_tokens_default: int = int(os.environ.get("THINKER_MAX_TOKENS_DEFAULT", "1024"))
    worker_cls: str = "server_thinker.TensorCapturingGPUWorker"

    def __post_init__(self):
        if self.limit_mm_per_prompt is None:
            object.__setattr__(self, "limit_mm_per_prompt", {"audio": 888, "image": 888})


CONFIG = ServerConfig()

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _align_cached_hiddens_to_token_ids(
    cached: list[tuple[int, torch.Tensor]],
    generated_token_ids: list[int],
) -> torch.Tensor:
    """Pick the contiguous slice of cached ``(sampled_token_id, hidden_row)`` whose
    sampled id sequence equals ``generated_token_ids`` (completion is ground truth).

    Each row ``i`` of the result is the **vLLM** ``sample_hidden_states`` row for the
    decode step that sampled ``generated_token_ids[i]``: that tensor is passed to
    ``compute_logits`` then ``sample_tokens`` (see ``gpu_model_runner.py``:
    ``sample_hidden_states = hidden_states[logits_indices]``). So row ``i`` is the
    hidden **used to predict** token ``i``, not the residual “at token *i* after
    consuming it”. Pairing that row with ``embed_input_ids(generated_token_ids[i])``
    is **wrong** for ``text_projection(emb) + hidden_projection(top)``; use
    :func:`_training_aligned_top_hidden_and_embed` instead.

    vLLM may run sampling steps whose tokens never appear in the final
    ``CompletionOutput.token_ids`` (scheduler trim, MM/prefill edge paths, etc.).
    Matching on token ids avoids guessing with discard masks or silent truncation.
    """
    if not generated_token_ids:
        if not cached:
            return torch.empty(0)
        raise RuntimeError(
            "Non-empty hidden cache but empty generated_token_ids; "
            f"cached_token_ids={[t for t, _ in cached]!r}"
        )
    if not cached:
        return torch.empty(0)

    cached_ids = [t for t, _ in cached]
    gen = generated_token_ids
    n, m = len(cached_ids), len(gen)
    if m > n:
        raise RuntimeError(
            "Fewer cached decode steps than completion tokens: "
            f"cached_len={n}, completion_len={m}, "
            f"cached_token_ids={cached_ids!r}, completion_token_ids={gen!r}"
        )

    match_starts: list[int] = []
    for start in range(n - m + 1):
        if cached_ids[start : start + m] == gen:
            match_starts.append(start)

    if not match_starts:
        raise RuntimeError(
            "completion.token_ids is not a contiguous subsequence of cached "
            f"per-step sample token ids: cached={cached_ids!r}, completion={gen!r}"
        )
    # If several windows match (repeated token patterns), prefer the last offset:
    # extra matching prefixes correspond to decode steps that never reached
    # ``CompletionOutput.token_ids`` in order.
    start = match_starts[-1]
    rows = [cached[i][1] for i in range(start, start + m)]
    return torch.cat(rows, dim=0)


def _training_aligned_top_hidden_and_embed(
    reply_hidden: torch.Tensor,
    text_embeds: torch.Tensor,
    generated_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Same **token-position** pairing as ms-swift / Megatron assistant slice (c.f. ``visible_end = end - 1``).

    Let completion length be ``n = len(generated_token_ids)``. Row ``reply_hidden[i]``
    is the hidden that **produces logits for** ``generated_token_ids[i]`` (pre–next-token
    causal position). Row ``text_embeds[k]`` is ``embed(generated_token_ids[k])``.

    For ``k = 0 .. n-2``, the training-consistent pair is ``text_embeds[k]`` with
    ``reply_hidden[k+1]`` (hidden after the prefix through token ``k``, used to predict
    token ``k+1`` — i.e. the same sequence slot as token ``k``'s **subsequent** residual
    path; equivalently: pair embed of token ``k`` with top hidden that sits at the
    boundary **after** token ``k`` in the autoregressive stack).

    The **last** completion token (index ``n-1``) has no following decode step that
    yields “its own” top hidden at streaming boundary; it is **omitted**, matching
    assistant ``assist[:-1]`` / ``visible_end = end - 1`` in training.

    Returns ``(top_aligned, emb_aligned, generated_token_ids[:-1])`` with length ``max(0, n-1)``.
    """
    n = len(generated_token_ids)
    if reply_hidden.shape[0] != n:
        raise RuntimeError(
            "reply_hidden rows must equal completion length before training alignment: "
            f"{reply_hidden.shape[0]} vs {n}"
        )
    if text_embeds.shape[0] != n:
        raise RuntimeError(
            "text_embeds rows must equal completion length before training alignment: "
            f"{text_embeds.shape[0]} vs {n}"
        )
    if n <= 1:
        return reply_hidden[:0], text_embeds[:0], []
    top = reply_hidden[1:]
    emb = text_embeds[:-1]
    aligned_ids = generated_token_ids[:-1]
    if top.shape[0] != emb.shape[0] or top.shape[0] != len(aligned_ids):
        raise RuntimeError(
            f"internal training-alignment shape mismatch: top={top.shape}, emb={emb.shape}, "
            f"ids={len(aligned_ids)}"
        )
    return top, emb, aligned_ids


class TensorCapturingGPUModelRunner(VllmGPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per external request id: list of (sampled_token_id, hidden row [1, dim])
        self.reply_hidden_state_cache: dict[str, list[tuple[int, torch.Tensor]]] = {}

    @staticmethod
    def to_external_req_id(req_id: str) -> str:
        return req_id.rsplit("-", 1)[0]

    def cache_step_reply_hidden_states(
        self,
        req_ids: list[str],
        sample_hidden_states: torch.Tensor,
        sampled_token_ids: list[list[int]],
    ) -> None:
        if not getattr(self, "rank", 0) == 0:
            return

        if sample_hidden_states.shape[0] != len(req_ids):
            raise RuntimeError(
                "sample_hidden_states rows do not match active req_ids: "
                f"{sample_hidden_states.shape[0]} vs {len(req_ids)}"
            )

        if len(sampled_token_ids) != len(req_ids):
            raise RuntimeError(
                "sampled_token_ids rows do not match active req_ids: "
                f"{len(sampled_token_ids)} vs {len(req_ids)}"
            )

        for req_id, hidden_state, token_ids in zip(
            req_ids, sample_hidden_states, sampled_token_ids
        ):
            if len(token_ids) == 0:
                continue
            if len(token_ids) != 1:
                raise RuntimeError(
                    "server_thinker.py expects one sampled token per request step, "
                    f"but got {len(token_ids)} for request {req_id}"
                )
            external_req_id = self.to_external_req_id(req_id)
            row = hidden_state.unsqueeze(0).contiguous().clone()
            self.reply_hidden_state_cache.setdefault(external_req_id, []).append(
                (int(token_ids[0]), row)
            )

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: Any | None) -> Any:
        req_ids: list[str] | None = None
        sample_hidden_states: torch.Tensor | None = None
        if self.execute_model_state is not None:
            req_ids = self.input_batch.req_ids.copy()
            sample_hidden_states = self.execute_model_state.sample_hidden_states.clone()

        output = super().sample_tokens(grammar_output)

        if req_ids is None or sample_hidden_states is None:
            return output

        if isinstance(output, AsyncGPUModelRunnerOutput):
            return TensorCapturingAsyncOutput(
                runner=self,
                req_ids=req_ids,
                sample_hidden_states=sample_hidden_states,
                inner=output,
            )

        if isinstance(output, ModelRunnerOutput):
            self.cache_step_reply_hidden_states(
                req_ids=req_ids,
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=output.sampled_token_ids,
            )
        return output


class TensorCapturingAsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        runner: TensorCapturingGPUModelRunner,
        req_ids: list[str],
        sample_hidden_states: torch.Tensor,
        inner: AsyncGPUModelRunnerOutput,
    ) -> None:
        self.runner = runner
        self.req_ids = req_ids
        self.sample_hidden_states = sample_hidden_states
        self.inner = inner

    def get_output(self) -> ModelRunnerOutput:
        output = self.inner.get_output()
        self.runner.cache_step_reply_hidden_states(
            req_ids=self.req_ids,
            sample_hidden_states=self.sample_hidden_states,
            sampled_token_ids=output.sampled_token_ids,
        )
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class TensorCapturingGPUWorker(VllmGPUWorker):
    """Captures reply tensors from the real decode path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_device(self):
        super().init_device()
        if self.use_v2_model_runner:
            raise RuntimeError("server_thinker.py currently supports only the V1 GPU model runner")
        self.model_runner = TensorCapturingGPUModelRunner(self.vllm_config, self.device)

    @torch.inference_mode()
    def extract_reply_tensors(
        self,
        request_id: str,
        generated_token_ids: list[int],
    ) -> dict[str, Any] | None:
        # Under tensor parallel, `embed_input_ids` uses sharded embedding / collectives;
        # every TP rank must execute the same forward. Previously only rank 0 ran it while
        # other ranks returned None immediately, which deadlocks NCCL / SHM broadcast
        # (symptom: shm_broadcast "No available shared memory broadcast block" + hang).

        if len(generated_token_ids) == 0:
            if self.rank != 0:
                return None
            return {
                "top_hidden_state": torch.empty(0),
                "text_embedding": torch.empty(0),
                "aligned_token_ids": [],
            }

        generated_ids = torch.tensor(
            generated_token_ids,
            dtype=torch.int32,
            device=self.model_runner.device,
        )
        model = self.get_model()
        if hasattr(model, "language_model") and hasattr(model.language_model, "embed_input_ids"):
            text_embeds = model.language_model.embed_input_ids(generated_ids)
        else:
            text_embeds = model.embed_input_ids(generated_ids)

        if self.rank != 0:
            return None

        cached_parts = self.model_runner.reply_hidden_state_cache.pop(request_id, [])
        if cached_parts:
            reply_hidden = _align_cached_hiddens_to_token_ids(
                cached_parts, generated_token_ids
            )
        else:
            reply_hidden = torch.empty(0, device=self.model_runner.device, dtype=self.model_runner.dtype)

        if reply_hidden.shape[0] != len(generated_token_ids):
            raise RuntimeError(
                "After token-id alignment, hidden rows still mismatch completion length: "
                f"{reply_hidden.shape[0]} vs {len(generated_token_ids)} for {request_id}"
            )

        top_out, emb_out, aligned_ids = _training_aligned_top_hidden_and_embed(
            reply_hidden, text_embeds, generated_token_ids
        )
        return {
            "top_hidden_state": top_out.to(dtype=torch.float16, device="cpu").contiguous(),
            "text_embedding": emb_out.to(dtype=torch.float16, device="cpu").contiguous(),
            "aligned_token_ids": aligned_ids,
        }


def tensor_debug_info(name: str, tensor: Any) -> dict[str, Any]:
    is_tensor = torch.is_tensor(tensor)
    is_fake_tensor = False
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        is_fake_tensor = isinstance(tensor, FakeTensor)
    except Exception:
        is_fake_tensor = False

    info: dict[str, Any] = {
        "name": name,
        "is_tensor": is_tensor,
        "python_type": f"{type(tensor).__module__}.{type(tensor).__name__}",
        "is_fake_tensor": is_fake_tensor,
    }
    if is_tensor:
        info.update(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": bool(tensor.requires_grad),
                "is_contiguous": bool(tensor.is_contiguous()),
                "numel": int(tensor.numel()),
            }
        )
    elif isinstance(tensor, (list, tuple)):
        info["is_nested_list"] = True
        info["outer_len"] = len(tensor)
        # vLLM MsgpackEncoder encodes torch.Tensor as (dtype_str, shape, data); see vllm/v1/serial_utils.py
        if (
            len(tensor) == 3
            and isinstance(tensor[0], str)
            and isinstance(tensor[1], (list, tuple))
        ):
            info["vllm_msgpack_tensor_triple"] = True
            info["dtype_str"] = tensor[0]
            info["shape"] = list(tensor[1])
            info["data_kind"] = type(tensor[2]).__name__
        elif tensor and isinstance(tensor[0], list):
            info["inner_len_row0"] = len(tensor[0])
    return info


def log_tensor_debug(name: str, tensor: Any) -> None:
    info = tensor_debug_info(name, tensor)
    LOGGER.info("Tensor check %s", info)


def _tensor_from_vllm_msgpack_triple(triple: Sequence[Any]) -> torch.Tensor:
    """Rebuild torch.Tensor from vLLM Msgpack wire form (must match vllm/v1/serial_utils.py).

    Encoder stores each tensor as ``(dtype_str, shape_tuple, data)`` where ``data`` is either
    an inline buffer or an index into aux buffers (only the EngineCore decoder can resolve the
    latter).
    """
    if len(triple) != 3:
        raise ValueError(f"vLLM tensor triple must have length 3, got {len(triple)}")
    dtype_str, shape, data = triple[0], triple[1], triple[2]
    if not isinstance(dtype_str, str):
        raise TypeError(f"triple[0] must be dtype str, got {type(dtype_str).__name__}")
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"triple[1] must be shape sequence, got {type(shape).__name__}")
    shape = tuple(int(x) for x in shape)
    if isinstance(data, int):
        raise RuntimeError(
            "Tensor triple carries zero-copy aux buffer index without MsgpackDecoder context; "
            "cannot reconstruct. See vllm/v1/serial_utils.py MsgpackDecoder._decode_tensor "
            "and VLLM_MSGPACK_ZERO_COPY_THRESHOLD."
        )
    try:
        torch_dtype = getattr(torch, dtype_str)
    except AttributeError:
        raise TypeError(f"unknown torch dtype in vLLM triple: {dtype_str!r}") from None
    if not isinstance(torch_dtype, torch.dtype):
        torch_dtype = torch.dtype(torch_dtype)

    if isinstance(data, (bytes, bytearray)):
        buffer = memoryview(data)
    elif isinstance(data, memoryview):
        buffer = data
    else:
        raw = getattr(data, "data", None)
        if isinstance(raw, (bytes, bytearray, memoryview)):
            buffer = memoryview(raw)
        else:
            try:
                buffer = memoryview(data)
            except TypeError as e:
                raise TypeError(
                    f"triple[2] must be bytes-like or msgpack.Ext after IPC decode; "
                    f"got {type(data).__name__}"
                ) from e

    if buffer.nbytes == 0:
        if shape and 0 not in shape:
            raise RuntimeError(f"empty tensor buffer but shape is {shape}")
        return torch.empty(shape, dtype=torch_dtype)

    arr_u8 = torch.frombuffer(buffer, dtype=torch.uint8)
    arr_u8 = arr_u8.clone()
    # Match vllm MsgpackDecoder._decode_tensor: view dtype then reshape to logical shape
    return arr_u8.view(torch_dtype).reshape(shape)


def _tensor_from_collective_ipc_value(obj: Any) -> torch.Tensor:
    """Decode value returned from ``collective_rpc`` on the API process: Tensor or vLLM triple."""
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, (list, tuple)) and len(obj) == 3 and isinstance(obj[0], str):
        return _tensor_from_vllm_msgpack_triple(obj)
    raise TypeError(
        "collective_rpc tensor field must be torch.Tensor or vLLM (dtype_str, shape, data) "
        f"triple; got {type(obj).__name__}"
    )


def _tensor_from_collective_ipc_value_cpu_float16(obj: Any) -> torch.Tensor:
    """Same as worker-side CPU float16 tensors used in responses and disk store."""
    t = _tensor_from_collective_ipc_value(obj)
    return t.to(dtype=torch.float16, device="cpu").contiguous()


class HiddenDiskStore:
    """Append-only per-request tensor shards on disk; enqueue from the hot path, IO in a thread."""

    def __init__(self, root_dir: str, queue_maxsize: int = 512) -> None:
        self.root = Path(root_dir)
        self.chunks_dir = self.root / "chunks"
        self.manifest_path = self.root / "manifest.jsonl"
        self._queue: Queue = Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_seq_lock = threading.Lock()
        self._next_seq = 1

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._next_seq = self._scan_next_seq()
        self._thread = threading.Thread(target=self._worker_loop, name="hidden-disk-store", daemon=True)
        self._thread.start()
        LOGGER.info("Hidden disk store enabled root=%s next_seq=%s", self.root, self._next_seq)

    def _scan_next_seq(self) -> int:
        mx = 0
        for p in self.chunks_dir.glob("req_*.pt"):
            try:
                mx = max(mx, int(p.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return mx + 1

    def submit_from_ipc(
        self,
        top_obj: Any,
        emb_obj: Any,
        aligned_token_ids: list[int],
        request_id: str,
        completion_token_ids_full: list[int] | None = None,
    ) -> float:
        """Convert IPC payloads, clone, enqueue. ``aligned_token_ids`` length must match tensor rows (``n-1`` pairs)."""
        t0 = time.perf_counter()
        top = _tensor_from_collective_ipc_value_cpu_float16(top_obj).clone()
        emb = _tensor_from_collective_ipc_value_cpu_float16(emb_obj).clone()
        if top.shape[0] != emb.shape[0]:
            raise RuntimeError(
                "top_hidden_state and text_embedding row counts differ after IPC decode: "
                f"{top.shape} vs {emb.shape} for {request_id}"
            )
        if top.shape[0] != len(aligned_token_ids):
            raise RuntimeError(
                "training-aligned tensor rows mismatch aligned_token_ids after IPC decode: "
                f"{top.shape[0]} vs {len(aligned_token_ids)} for {request_id}"
            )
        try:
            self._queue.put_nowait(
                (top, emb, list(aligned_token_ids), request_id, completion_token_ids_full)
            )
        except Full:
            LOGGER.warning(
                "hidden disk store queue full (maxsize=%d), drop request_id=%s",
                self._queue.maxsize,
                request_id,
            )
        return time.perf_counter() - t0

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                LOGGER.warning("hidden disk store worker thread did not exit cleanly")

    def _persist_one(
        self,
        top: torch.Tensor,
        emb: torch.Tensor,
        token_ids: list[int],
        request_id: str,
        completion_token_ids_full: list[int] | None,
    ) -> None:
        with self._next_seq_lock:
            seq = self._next_seq
            self._next_seq += 1
        path = self.chunks_dir / f"req_{seq:08d}.pt"
        # One HTTP completion = one assistant turn here; each turn is an explicit (top, emb) pair
        payload = {
            "request_id": request_id,
            # Per-row ids for ``top_hidden_state[j]`` / ``text_embedding[j]`` (training-aligned; length n-1)
            "generated_token_ids": token_ids,
            "completion_token_ids_full": completion_token_ids_full,
            "seq": seq,
            "turns": [
                {
                    "turn_index": 0,
                    "top_hidden_state": top,
                    "text_embedding": emb,
                }
            ],
        }
        t_io = time.perf_counter()
        try:
            torch.save(payload, path)
        except Exception:
            LOGGER.exception("hidden disk store failed seq=%s request_id=%s path=%s", seq, request_id, path)
            return
        io_s = time.perf_counter() - t_io
        meta: dict[str, Any] = {
            "seq": seq,
            "request_id": request_id,
            "path": str(path.relative_to(self.root)),
            "top_shape": list(top.shape),
            "emb_shape": list(emb.shape),
            "n_training_aligned_rows": len(token_ids),
            "save_io_s": round(io_s, 6),
            "saved_at_unix": int(time.time()),
        }
        if completion_token_ids_full is not None:
            meta["completion_tokens_full"] = len(completion_token_ids_full)
        try:
            with self.manifest_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.exception("hidden disk store manifest append failed seq=%s", seq)
        LOGGER.debug(
            "hidden_disk_save seq=%s save_io_s=%.6f top_shape=%s emb_shape=%s",
            seq,
            io_s,
            list(top.shape),
            list(emb.shape),
        )

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                if self._stop.is_set():
                    while True:
                        try:
                            pending = self._queue.get_nowait()
                        except Empty:
                            return
                        if pending is not None:
                            top, emb, token_ids, rid, full_ids = pending
                            self._persist_one(top, emb, token_ids, rid, full_ids)
                continue
            if item is None:
                return
            top, emb, token_ids, request_id, full_ids = item
            self._persist_one(top, emb, token_ids, request_id, full_ids)


def _strip_data_uri(data: str) -> str:
    if data.startswith("data:"):
        return data.split(",", 1)[-1]
    return data


def _pcm_bytes_to_float32(
    pcm: bytes,
    *,
    sample_width: int,
    channels: int,
) -> np.ndarray:
    if sample_width == 1:
        audio = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported WAV sample width: {sample_width} bytes.")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _decode_base64_audio(data: str) -> bytes:
    try:
        return base64.b64decode(_strip_data_uri(data), validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio payload: expected base64-encoded WAV bytes or a local file path.",
        ) from exc


def _decode_audio_bytes(data: bytes, sample_rate: float | None) -> tuple[np.ndarray, float | None]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            wav_sample_rate = float(wav_file.getframerate())
            sample_width = wav_file.getsampwidth()
            pcm = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio payload: only WAV container bytes are accepted.",
        ) from exc

    audio = _pcm_bytes_to_float32(pcm, sample_width=sample_width, channels=channels)
    return audio, sample_rate if sample_rate is not None else wav_sample_rate


def _normalize_audio_payload(payload: Any) -> tuple[np.ndarray, float | None]:
    sample_rate: float | None = None

    if isinstance(payload, dict):
        raw_sample_rate = payload.get("sample_rate") or payload.get("sampling_rate")
        if raw_sample_rate is not None:
            try:
                sample_rate = float(raw_sample_rate)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid audio sample rate: {raw_sample_rate!r}") from exc

        encoded = payload.get("data") or payload.get("bytes")
        if encoded is not None:
            if isinstance(encoded, bytes):
                return _decode_audio_bytes(encoded, sample_rate)
            if isinstance(encoded, str):
                return _decode_audio_bytes(_decode_base64_audio(encoded), sample_rate)
            raise HTTPException(status_code=400, detail="Unsupported audio payload encoding.")

        payload = payload.get("url") or payload.get("path")

    if isinstance(payload, bytes):
        return _decode_audio_bytes(payload, sample_rate)

    if isinstance(payload, str):
        candidate_path = Path(payload)
        if candidate_path.exists():
            return _decode_audio_bytes(candidate_path.read_bytes(), sample_rate)
        return _decode_audio_bytes(_decode_base64_audio(payload), sample_rate)

    raise HTTPException(status_code=400, detail="Unsupported audio payload.")


def _extract_audio_item(item: dict[str, Any]) -> tuple[np.ndarray, float | None]:
    item_type = item.get("type")
    if item_type == "input_audio":
        payload = item.get("input_audio", {})
        if not isinstance(payload, dict) or "data" not in payload:
            raise HTTPException(status_code=400, detail="input_audio.data is required.")
        return _normalize_audio_payload(payload)

    payload = item.get("audio") or item.get("audio_url")
    return _normalize_audio_payload(payload)


def _extract_visual_item(item: dict[str, Any], media_key: str) -> Any:
    payload = None
    for key in (media_key, f"{media_key}_url", f"input_{media_key}"):
        if key in item:
            payload = item.get(key)
            break
    if isinstance(payload, dict):
        payload = payload.get("url") or payload.get("data") or payload
    return payload


def _normalize_content_parts(
    content: Any,
    audios: list[Any],
    images: list[Any],
    videos: list[Any],
) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise HTTPException(
            status_code=400,
            detail="message.content must be either a string or a content-part list",
        )

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="each content part must be an object")

        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(item.get("text", ""))
        elif item_type in {"input_audio", "audio", "audio_url"}:
            audios.append(_extract_audio_item(item))
            text_parts.append(AUDIO_PLACEHOLDER)
        elif item_type in {"image", "image_url", "input_image"}:
            images.append(_extract_visual_item(item, "image"))
            text_parts.append(IMAGE_PLACEHOLDER)
        elif item_type in {"video", "video_url", "input_video"}:
            videos.append(_extract_visual_item(item, "video"))
            text_parts.append(VIDEO_PLACEHOLDER)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported content part type: {item_type!r}; supported types: "
                    "text, image_url, audio_url, video_url, input_audio"
                ),
            )
    return "".join(text_parts)


def build_engine_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    conversation_id: str | None,
) -> dict[str, Any]:
    prompt_messages = []
    audios: list[Any] = []
    images: list[Any] = []
    videos: list[Any] = []
    for message in messages:
        if "role" not in message or "content" not in message:
            raise HTTPException(status_code=400, detail="Each message needs role/content")
        normalized_message = {
            "role": message["role"],
            "content": _normalize_content_parts(message["content"], audios, images, videos),
        }
        for optional_field in ("name", "tool_call_id", "tool_calls"):
            if optional_field in message:
                normalized_message[optional_field] = message[optional_field]
        prompt_messages.append(normalized_message)

    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    engine_prompt: dict[str, Any] = {"prompt": prompt}
    if conversation_id:
        engine_prompt["cache_salt"] = conversation_id
    multimodal_data: dict[str, Any] = {}
    if audios:
        multimodal_data["audio"] = audios
    if images:
        multimodal_data["image"] = images
    if videos:
        multimodal_data["video"] = videos
    if multimodal_data:
        engine_prompt["multi_modal_data"] = multimodal_data
    return engine_prompt


def build_sampling_params(payload: dict[str, Any]) -> SamplingParams:
    max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or CONFIG.max_tokens_default)
    temperature = float(payload.get("temperature", 0.0))
    top_p = float(payload.get("top_p", 1.0))
    stop = payload.get("stop")
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )


def _resolve_session_id(payload: dict[str, Any]) -> str | None:
    raw = payload.get("session_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    vx = payload.get("vllm_xargs")
    if isinstance(vx, dict):
        conv = vx.get("conversation_id")
        if isinstance(conv, str) and conv.strip():
            return conv.strip()
    raw = payload.get("conversation_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _build_openai_response(
    request_id: str,
    created: int,
    completion: Any,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": CONFIG.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion.text,
                },
                "finish_reason": completion.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(generated_token_ids),
            "total_tokens": len(prompt_token_ids) + len(generated_token_ids),
        },
    }


async def _run_chat_turn(
    payload: dict[str, Any],
    *,
    persist_hidden: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if payload.get("stream"):
        raise HTTPException(status_code=400, detail="stream=true is not supported")

    model_name = payload.get("model")
    if model_name is not None and model_name != CONFIG.served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"model must be exactly {CONFIG.served_model_name}",
        )

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    engine: AsyncLLM = app.state.engine
    processor = app.state.processor
    sampling_params = build_sampling_params(payload)
    session_id = _resolve_session_id(payload)
    engine_prompt = build_engine_prompt(
        processor,
        messages,
        conversation_id=session_id,
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    t_engine_start = time.perf_counter()
    async with app.state.request_lock:
        final_output = None
        async for output in engine.generate(
            prompt=engine_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="vLLM returned no output")

        completion = final_output.outputs[0]
        generated_token_ids = list(completion.token_ids)
        prompt_token_ids = list(final_output.prompt_token_ids or [])
        if not prompt_token_ids:
            raise HTTPException(
                status_code=500,
                detail="vLLM did not return prompt_token_ids for tensor extraction",
            )
        tensor_results = await engine.collective_rpc(
            "extract_reply_tensors",
            args=(request_id, generated_token_ids),
        )
        tensor_payload = next((item for item in tensor_results if item is not None), None)
        if tensor_payload is None:
            raise HTTPException(status_code=500, detail="failed to extract reply tensors")
        log_tensor_debug("top_hidden_state", tensor_payload["top_hidden_state"])
        log_tensor_debug("text_embedding", tensor_payload["text_embedding"])

        aligned_token_ids = list(tensor_payload.get("aligned_token_ids") or [])
        top_cpu = _tensor_from_collective_ipc_value_cpu_float16(tensor_payload["top_hidden_state"])
        emb_cpu = _tensor_from_collective_ipc_value_cpu_float16(tensor_payload["text_embedding"])

        hidden_save_enqueue_s = 0.0
        hs = getattr(app.state, "hidden_store", None)
        if persist_hidden and hs is not None:
            hidden_save_enqueue_s = hs.submit_from_ipc(
                tensor_payload["top_hidden_state"],
                tensor_payload["text_embedding"],
                aligned_token_ids,
                request_id,
                completion_token_ids_full=generated_token_ids,
            )

    engine_and_rpc_s = time.perf_counter() - t_engine_start
    created = int(time.time())
    response = _build_openai_response(
        request_id=request_id,
        created=created,
        completion=completion,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
    )
    internal_payload = {
        "session_id": session_id,
        "request_id": request_id,
        "generated_token_ids": aligned_token_ids,
        "completion_token_ids_full": generated_token_ids,
        "response": response,
        "text": completion.text,
        "turns": [
            {
                "turn_index": 0,
                "top_hidden_state": top_cpu,
                "text_embedding": emb_cpu,
            }
        ],
    }
    metrics = {
        "engine_generate_plus_collective_rpc_s": engine_and_rpc_s,
        "hidden_save_enqueue_s": hidden_save_enqueue_s,
        "completion_tokens": len(generated_token_ids),
        "training_aligned_rows": len(aligned_token_ids),
        "request_id": request_id,
    }
    return response, internal_payload, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_args = AsyncEngineArgs(
        model=CONFIG.model,
        served_model_name=CONFIG.served_model_name,
        tensor_parallel_size=CONFIG.tensor_parallel_size,
        max_model_len=CONFIG.max_model_len,
        enable_prefix_caching=CONFIG.enable_prefix_caching,
        limit_mm_per_prompt=CONFIG.limit_mm_per_prompt,
        worker_cls=CONFIG.worker_cls,
        gpu_memory_utilization=CONFIG.gpu_memory_utilization,
        dtype=CONFIG.dtype,
    )
    app.state.engine = AsyncLLM.from_engine_args(engine_args)
    app.state.processor = AutoProcessor.from_pretrained(
        CONFIG.model,
        trust_remote_code=True,
    )
    app.state.request_lock = asyncio.Lock()
    raw_store = os.environ.get("THINKER_HIDDEN_STORE_DIR")
    if raw_store is not None and raw_store.strip().lower() in ("", "0", "false", "no", "off"):
        app.state.hidden_store = None
    else:
        store_dir = (
            raw_store.strip()
            if raw_store is not None and raw_store.strip()
            else os.path.join(ROOT_DIR, "thinker_hidden_store")
        )
        app.state.hidden_store = HiddenDiskStore(store_dir)
        app.state.hidden_store.start()
    try:
        yield
    finally:
        hs = getattr(app.state, "hidden_store", None)
        if hs is not None:
            hs.shutdown()
        engine = app.state.engine
        if engine is not None:
            engine.shutdown()


app = FastAPI(title="realtime_serving_talker", lifespan=lifespan)
app.state.engine = None
app.state.processor = None
app.state.request_lock = None
app.state.hidden_store = None


@app.get("/health")
async def health() -> dict[str, Any]:
    hs = getattr(app.state, "hidden_store", None)
    return {
        "ok": True,
        "model": CONFIG.served_model_name,
        "port": CONFIG.port,
        "tensor_parallel_size": CONFIG.tensor_parallel_size,
        "enable_prefix_caching": CONFIG.enable_prefix_caching,
        "limit_mm_per_prompt": CONFIG.limit_mm_per_prompt,
        "max_model_len": CONFIG.max_model_len,
        "thinker_hidden_store_dir": str(hs.root) if hs is not None else None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    t_handler_start = time.perf_counter()
    t_json_start = time.perf_counter()
    payload = await request.json()
    json_deserialize_s = time.perf_counter() - t_json_start
    response, _internal_payload, metrics = await _run_chat_turn(payload, persist_hidden=True)
    handler_total_s = time.perf_counter() - t_handler_start
    LOGGER.info(
        "chat_completions timing request_id=%s json_deserialize_s=%.6f "
        "engine_generate_plus_collective_rpc_s=%.6f hidden_save_enqueue_s=%.6f "
        "handler_total_s=%.6f completion_tokens=%d training_aligned_rows=%d "
        "(row k: text_embedding[k]=embed(token_k), top_hidden[k]=reply_hidden[k+1] "
        "vLLM sample_hidden; last completion token omitted like assist[:-1]; "
        "disk IO async, manifest.jsonl save_io_s)",
        metrics["request_id"],
        json_deserialize_s,
        metrics["engine_generate_plus_collective_rpc_s"],
        metrics["hidden_save_enqueue_s"],
        handler_total_s,
        metrics["completion_tokens"],
        metrics["training_aligned_rows"],
    )
    return JSONResponse(response)


@app.post("/internal/chat_turn")
async def internal_chat_turn(request: Request) -> Response:
    t_handler_start = time.perf_counter()
    t_json_start = time.perf_counter()
    payload = await request.json()
    json_deserialize_s = time.perf_counter() - t_json_start
    response, internal_payload, metrics = await _run_chat_turn(payload, persist_hidden=False)
    handler_total_s = time.perf_counter() - t_handler_start
    LOGGER.info(
        "internal_chat_turn timing request_id=%s session_id=%s json_deserialize_s=%.6f "
        "engine_generate_plus_collective_rpc_s=%.6f handler_total_s=%.6f "
        "completion_tokens=%d training_aligned_rows=%d",
        metrics["request_id"],
        internal_payload.get("session_id"),
        json_deserialize_s,
        metrics["engine_generate_plus_collective_rpc_s"],
        handler_total_s,
        metrics["completion_tokens"],
        metrics["training_aligned_rows"],
    )
    bio = io.BytesIO()
    torch.save(internal_payload, bio)
    return Response(
        content=bio.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Thinker-Request-Id": str(metrics["request_id"]),
            "X-Thinker-Session-Id": str(internal_payload.get("session_id") or ""),
            "X-Thinker-Text-Len": str(len(str(internal_payload.get("text") or ""))),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port, log_level="info")
