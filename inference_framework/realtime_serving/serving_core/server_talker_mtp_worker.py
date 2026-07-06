"""
Dedicated MTP vLLM subprocess entry (``spawn``).

Parent sets ``CUDA_VISIBLE_DEVICES`` for this process (default **``0``**：与 MoE TP=4 的 rank0 **同物理卡**，
显存由 ``TALKER_MTP_GPU_MEM_UTIL`` 与父进程 ``TALKER_GPU_MEM_UTIL`` 分账)。The child loads a **full** talker checkpoint (same class as MoE engine) so
``LLM.apply_model(vllm_mtp_run_rpc)`` runs in an isolated executor; warmup shows its own
``Capturing CUDA graphs`` in the shared terminal (stdout/stderr inherited).

See ``TALKER_MTP_SPLIT_ENGINE`` / ``TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES`` in
``server_talker.py``.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from functools import partial
from multiprocessing.connection import Connection
from typing import Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_ROOT = os.environ.get(
    "VLLM_ROOT",
    os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "vllm_qwen3_omni")),
)

_LOG = logging.getLogger("talker_mtp_worker")


def _ensure_paths_and_env() -> None:
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    if VLLM_ROOT not in sys.path:
        sys.path.insert(0, VLLM_ROOT)
    _pp = os.environ.get("PYTHONPATH", "")
    _pp_parts = [p for p in _pp.split(os.pathsep) if p]
    if ROOT_DIR not in _pp_parts:
        os.environ["PYTHONPATH"] = ROOT_DIR if not _pp else ROOT_DIR + os.pathsep + _pp
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") is None:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ.setdefault("TALKER_MTP_VERBOSE_LOG", "1")
    os.environ.setdefault("TALKER_MTP_WORKER_VERBOSE", "1")
    os.environ.setdefault("TALKER_MTP_PROFILE", "1")


def mtp_worker_main(conn: Connection, cuda_visible: str) -> None:
    """Child: load vLLM once, then serve ``mtp`` ops until ``shutdown``.

    ``multiprocessing.Process`` **没有** ``env=`` 参数（那是 ``subprocess.Popen`` 的 API）；
    必须在 **本函数第一行** 设置 ``CUDA_VISIBLE_DEVICES``，再导入任何会初始化 CUDA 的模块。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    _ensure_paths_and_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _LOG.info(
        "MTP subprocess start pid=%s CUDA_VISIBLE_DEVICES=%r (isolated engine; own graph capture)",
        os.getpid(),
        cuda_visible,
    )
    try:
        from server_talker import _talker_hf_overrides
        from server_talker_rpc import vllm_mtp_run_rpc
        from vllm.entrypoints.llm import LLM
    except Exception as e:
        conn.send({"stage": "error", "err": f"import failed: {e}", "tb": traceback.format_exc()})
        raise

    model = os.environ.get("TALKER_MODEL", "").strip()
    if not model:
        msg = {"stage": "error", "err": "TALKER_MODEL empty", "tb": ""}
        conn.send(msg)
        raise RuntimeError(msg["err"])

    tp = int(os.environ.get("TALKER_MTP_SUBPROC_TP", os.environ.get("TALKER_MTP_TP", "1")))
    max_model_len = int(
        os.environ.get(
            "TALKER_MTP_SUBPROC_MAX_MODEL_LEN",
            os.environ.get("TALKER_MAX_MODEL_LEN", "32386"),
        )
    )
    gpu_util = float(os.environ.get("TALKER_MTP_GPU_MEM_UTIL", "0.1"))
    prefix_cache = os.environ.get("TALKER_MTP_PREFIX_CACHE", os.environ.get("TALKER_PREFIX_CACHE", "1")) == "1"
    enforce_eager = os.environ.get("TALKER_MTP_ENFORCE_EAGER", os.environ.get("TALKER_ENFORCE_EAGER", "0")) == "1"

    conn.send({"stage": "loading", "tp": tp, "max_model_len": max_model_len})
    _LOG.info(
        "======== MTP 专用 vLLM 引擎开始加载（下面会出现 **本进程** 的 Loading safetensors / Capturing CUDA graphs，"
        "与父进程 MoE 那段是 **两套**）========"
    )
    try:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            hf_overrides=_talker_hf_overrides,
            enable_prompt_embeds=True,
            skip_tokenizer_init=True,
            max_model_len=max_model_len,
            tensor_parallel_size=tp,
            dtype="auto",
            gpu_memory_utilization=gpu_util,
            enforce_eager=enforce_eager,
            enable_prefix_caching=prefix_cache,
        )
    except Exception as e:
        conn.send({"stage": "error", "err": str(e), "tb": traceback.format_exc()})
        raise

    conn.send({"stage": "ready", "pid": os.getpid(), "tp": tp})
    _LOG.info(
        "======== MTP 专用 vLLM 引擎就绪 pid=%s tp=%s（上方 Capturing CUDA graphs 含 code_predictor 所在整图路径）========",
        os.getpid(),
        tp,
    )

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg == "shutdown":
            break
        if not isinstance(msg, dict) or msg.get("op") != "mtp":
            try:
                conn.send({"ok": False, "err": f"bad message: {msg!r}", "tb": ""})
            except Exception:
                pass
            continue
        try:
            outs = llm.apply_model(
                partial(
                    vllm_mtp_run_rpc,
                    last_talker_hidden=msg["last_h"],
                    layer0_token_id=int(msg["layer0_token_id"]),
                    temperature=float(msg["temperature"]),
                    top_k=int(msg["top_k"]),
                    top_p=float(msg["top_p"]),
                )
            )
            conn.send({"ok": True, "outs": outs})
        except Exception as e:
            try:
                conn.send({"ok": False, "err": str(e), "tb": traceback.format_exc()})
            except Exception:
                pass

    _LOG.info("MTP subprocess exit pid=%s", os.getpid())
