import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("server_orchestrator")


@dataclass(frozen=True)
class OrchestratorConfig:
    host: str = os.environ.get("ORCH_HOST", "0.0.0.0")
    port: int = int(os.environ.get("ORCH_PORT", "21000"))
    thinker_internal_url: str = os.environ.get(
        "THINKER_INTERNAL_URL",
        "http://127.0.0.1:19999/internal/chat_turn",
    )
    talker_internal_url_base: str = os.environ.get(
        "TALKER_INTERNAL_URL_BASE",
        "http://127.0.0.1:20000/internal/talker/turn",
    )
    talker_delete_url_base: str = os.environ.get(
        "TALKER_DELETE_URL_BASE",
        "http://127.0.0.1:20000/v1/talker/session",
    )
    request_timeout_s: float = float(os.environ.get("ORCH_REQUEST_TIMEOUT_S", "3600"))


CONFIG = OrchestratorConfig()


@dataclass
class SessionRuntime:
    session_id: str
    audio_queue: asyncio.Queue[bytes | None]
    talker_input_queue: asyncio.Queue[bytes | None]
    worker_task: asyncio.Task[Any]


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


def _post_thinker_turn(payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        CONFIG.thinker_internal_url,
        json=payload,
        timeout=CONFIG.request_timeout_s,
    )


def _post_talker_turn(session_id: str, blob: bytes) -> requests.Response:
    return requests.post(
        f"{CONFIG.talker_internal_url_base}/{session_id}",
        data=blob,
        headers={"content-type": "application/octet-stream"},
        timeout=CONFIG.request_timeout_s,
    )


def _delete_talker_session(session_id: str) -> requests.Response:
    return requests.delete(
        f"{CONFIG.talker_delete_url_base}/{session_id}",
        timeout=CONFIG.request_timeout_s,
    )


async def _close_runtime(runtime: SessionRuntime | None) -> None:
    if runtime is None:
        return
    try:
        await runtime.talker_input_queue.put(None)
    except Exception:
        pass
    try:
        await runtime.audio_queue.put(None)
    except Exception:
        pass
    try:
        await runtime.worker_task
    except Exception:
        LOGGER.exception("talker worker exit with error session_id=%s", runtime.session_id)
    try:
        await asyncio.to_thread(_delete_talker_session, runtime.session_id)
    except Exception:
        LOGGER.exception("failed to delete talker session session_id=%s", runtime.session_id)


async def _talker_worker(
    session_id: str,
    talker_input_queue: asyncio.Queue[bytes | None],
    audio_queue: asyncio.Queue[bytes | None],
) -> None:
    while True:
        blob = await talker_input_queue.get()
        if blob is None:
            return
        try:
            resp = await asyncio.to_thread(_post_talker_turn, session_id, blob)
        except Exception:
            LOGGER.exception("talker internal request failed session_id=%s", session_id)
            continue
        if resp.status_code == 204:
            LOGGER.info("talker produced no audio session_id=%s", session_id)
            continue
        if resp.status_code != 200:
            LOGGER.error(
                "talker internal bad status session_id=%s status=%s body=%s",
                session_id,
                resp.status_code,
                resp.text[:500],
            )
            continue
        if resp.content:
            await audio_queue.put(resp.content)


async def _ensure_runtime(session_id: str) -> SessionRuntime:
    runtime: SessionRuntime | None = getattr(app.state, "runtime", None)
    if runtime is not None and runtime.session_id != session_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"single-session orchestrator currently active for {runtime.session_id}; "
                f"close it before starting {session_id}"
            ),
        )
    if runtime is not None:
        return runtime
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    talker_input_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    worker_task = asyncio.create_task(
        _talker_worker(session_id, talker_input_queue, audio_queue)
    )
    runtime = SessionRuntime(
        session_id=session_id,
        audio_queue=audio_queue,
        talker_input_queue=talker_input_queue,
        worker_task=worker_task,
    )
    app.state.runtime = runtime
    LOGGER.info("orchestrator session started session_id=%s", session_id)
    return runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = None
    try:
        yield
    finally:
        await _close_runtime(getattr(app.state, "runtime", None))
        app.state.runtime = None


app = FastAPI(title="thinker_talker_orchestrator", lifespan=lifespan)
app.state.runtime = None


@app.get("/health")
async def health() -> dict[str, Any]:
    runtime: SessionRuntime | None = getattr(app.state, "runtime", None)
    return {
        "ok": True,
        "thinker_internal_url": CONFIG.thinker_internal_url,
        "talker_internal_url_base": CONFIG.talker_internal_url_base,
        "active_session_id": runtime.session_id if runtime is not None else None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id = _resolve_session_id(payload)
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required (payload.session_id or vllm_xargs.conversation_id)",
        )
    runtime = await _ensure_runtime(session_id)
    thinker_payload = dict(payload)
    thinker_payload["session_id"] = session_id
    thinker_payload.setdefault("vllm_xargs", {})
    if isinstance(thinker_payload["vllm_xargs"], dict):
        thinker_payload["vllm_xargs"].setdefault("conversation_id", session_id)

    t0 = asyncio.get_running_loop().time()
    try:
        resp = await asyncio.to_thread(_post_thinker_turn, thinker_payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"thinker internal request failed: {e}") from e
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"thinker internal bad status={resp.status_code}: {resp.text[:500]}",
        )
    try:
        internal_payload = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to decode thinker payload: {e}") from e
    response = internal_payload.get("response")
    if not isinstance(response, dict):
        raise HTTPException(status_code=502, detail="thinker payload missing response dict")

    talker_blob = io.BytesIO()
    torch.save(
        {
            "request_id": internal_payload.get("request_id"),
            "generated_token_ids": internal_payload.get("generated_token_ids"),
            "completion_token_ids_full": internal_payload.get("completion_token_ids_full"),
            "turns": internal_payload.get("turns") or [],
        },
        talker_blob,
    )
    await runtime.talker_input_queue.put(talker_blob.getvalue())
    LOGGER.info(
        "orchestrator chat turn session_id=%s thinker_http_s=%.6f text_len=%d",
        session_id,
        asyncio.get_running_loop().time() - t0,
        len(str(internal_payload.get("text") or "")),
    )
    return JSONResponse(response)


@app.websocket("/v1/audio/stream/{session_id}")
async def audio_stream(websocket: WebSocket, session_id: str) -> None:
    runtime = await _ensure_runtime(session_id)
    await websocket.accept()
    LOGGER.info("audio ws connected session_id=%s", session_id)
    try:
        while True:
            item = await runtime.audio_queue.get()
            if item is None:
                return
            await websocket.send_bytes(item)
    except WebSocketDisconnect:
        LOGGER.info("audio ws disconnected session_id=%s", session_id)


@app.delete("/v1/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, Any]:
    runtime: SessionRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None or runtime.session_id != session_id:
        return {"ok": True, "session_id": session_id, "cleared": False}
    await _close_runtime(runtime)
    app.state.runtime = None
    LOGGER.info("orchestrator session cleared session_id=%s", session_id)
    return {"ok": True, "session_id": session_id, "cleared": True}


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port, log_level="info")
