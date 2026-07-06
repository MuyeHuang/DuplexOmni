import argparse
import ast
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, **kwargs):
        return iterable


HM = os.environ.get("VOICEAGENT_ROOT", ".")
INPUT_DIR = f"{HM}/api_generate/pipeline_outputs_parquet/training_v7_codec"
OUTPUT_DIR = f"{HM}/api_generate/pipeline_outputs_parquet/training_v7_codec_noself"

DEFAULT_FILE_WORKERS = 8
DEFAULT_BATCH_ROWS = 128
DEFAULT_READ_QUEUE_BATCHES = 4
DEFAULT_WRITE_QUEUE_BATCHES = 4
DEFAULT_COMPRESSION = "snappy"


def strip_self_audio_from_user_content(content: str) -> str:
    if "self_audio" not in content:
        return content
    d = ast.literal_eval(content)
    if not isinstance(d, dict):
        return content
    if d.get("self_audio") == "<audio>":
        d = {k: v for k, v in d.items() if k != "self_audio"}
    return str(d)


def transform_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    audios = list(row["audios"]) if row.get("audios") is not None else []
    row["audios"] = audios[::2]

    messages = list(row["messages"]) if row.get("messages") is not None else []
    transformed_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            transformed_messages.append(msg)
            continue
        msg = dict(msg)
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                try:
                    msg["content"] = strip_self_audio_from_user_content(content)
                except (ValueError, SyntaxError):
                    pass
        transformed_messages.append(msg)
    row["messages"] = transformed_messages
    return row


def _debug_summary(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    before_audios = list(before.get("audios") or [])
    after_audios = list(after.get("audios") or [])
    before_users = [
        m for m in (before.get("messages") or [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    after_users = [
        m for m in (after.get("messages") or [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    before_content = before_users[0].get("content", "")[:160] if before_users else ""
    after_content = after_users[0].get("content", "")[:160] if after_users else ""
    return (
        f"audios {len(before_audios)} -> {len(after_audios)}; "
        f"user before={before_content!r}; user after={after_content!r}"
    )


def _put_with_backpressure(
    q: "queue.Queue",
    item: Any,
    stop_event: threading.Event,
    error_lists: List[List[BaseException]],
) -> bool:
    while True:
        if stop_event.is_set() and item is not None:
            return False
        for errors in error_lists:
            if errors:
                raise errors[0]
        try:
            q.put(item, timeout=1)
            return True
        except queue.Full:
            continue


def _put_sentinel(q: "queue.Queue", stop_event: threading.Event) -> None:
    while True:
        try:
            q.put(None, timeout=1)
            return
        except queue.Full:
            if stop_event.is_set():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


def process_parquet_file(
    input_path: str,
    output_path: str,
    batch_rows: int,
    read_queue_batches: int,
    write_queue_batches: int,
    compression: str,
    debug: bool = False,
) -> Dict[str, Any]:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    tmp_path = Path(str(output_path_obj) + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    start = time.time()
    pfile = pq.ParquetFile(str(input_path_obj))
    schema = pfile.schema_arrow
    read_queue: "queue.Queue[Optional[pa.RecordBatch]]" = queue.Queue(
        maxsize=max(1, int(read_queue_batches))
    )
    write_queue: "queue.Queue[Optional[pa.Table]]" = queue.Queue(
        maxsize=max(1, int(write_queue_batches))
    )
    stop_event = threading.Event()
    reader_errors: List[BaseException] = []
    writer_errors: List[BaseException] = []
    stats = {"rows_written": 0, "batches_written": 0}

    def reader_loop() -> None:
        try:
            local_pfile = pq.ParquetFile(str(input_path_obj))
            for batch in local_pfile.iter_batches(batch_size=max(1, int(batch_rows))):
                if not _put_with_backpressure(read_queue, batch, stop_event, [writer_errors]):
                    break
        except BaseException as exc:
            reader_errors.append(exc)
            stop_event.set()
        finally:
            _put_sentinel(read_queue, stop_event)

    def writer_loop() -> None:
        writer = None
        try:
            while True:
                table = write_queue.get()
                if table is None:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(tmp_path),
                        schema=schema,
                        compression=compression,
                    )
                writer.write_table(table, row_group_size=max(1, table.num_rows))
                stats["rows_written"] += table.num_rows
                stats["batches_written"] += 1
        except BaseException as exc:
            writer_errors.append(exc)
            stop_event.set()
        finally:
            if writer is not None:
                writer.close()

    reader_thread = threading.Thread(target=reader_loop, name=f"reader-{input_path_obj.name}", daemon=True)
    writer_thread = threading.Thread(target=writer_loop, name=f"writer-{input_path_obj.name}", daemon=True)
    reader_thread.start()
    writer_thread.start()

    rows_read = 0
    batches_read = 0
    debug_text = ""
    try:
        while True:
            if writer_errors:
                raise writer_errors[0]
            item = read_queue.get()
            if item is None:
                break
            rows = item.to_pylist()
            rows_read += len(rows)
            batches_read += 1
            transformed = [transform_row(row) for row in rows]
            if debug and not debug_text and rows and transformed:
                debug_text = _debug_summary(rows[0], transformed[0])
            table = pa.Table.from_pylist(transformed, schema=schema)
            _put_with_backpressure(write_queue, table, stop_event, [writer_errors])

        reader_thread.join()
        if reader_errors:
            raise reader_errors[0]
        if rows_read == 0:
            empty = pa.Table.from_pylist([], schema=schema)
            _put_with_backpressure(write_queue, empty, stop_event, [writer_errors])
        _put_sentinel(write_queue, stop_event)
        writer_thread.join()
        if writer_errors:
            raise writer_errors[0]
        os.replace(str(tmp_path), str(output_path_obj))
    except BaseException:
        stop_event.set()
        _put_sentinel(write_queue, stop_event)
        writer_thread.join(timeout=5)
        reader_thread.join(timeout=5)
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "input": input_path_obj.name,
        "output": output_path_obj.name,
        "rows_read": rows_read,
        "rows_written": stats["rows_written"],
        "batches_read": batches_read,
        "batches_written": stats["batches_written"],
        "seconds": time.time() - start,
        "debug": debug_text,
    }


def _submit_file(
    executor: ProcessPoolExecutor,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
):
    return executor.submit(
        process_parquet_file,
        str(input_path),
        str(output_path),
        args.batch_rows,
        args.read_queue_batches,
        args.write_queue_batches,
        args.compression,
        args.debug,
    )


def _process_one_serial(input_path: Path, output_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return process_parquet_file(
        str(input_path),
        str(output_path),
        args.batch_rows,
        args.read_queue_batches,
        args.write_queue_batches,
        args.compression,
        args.debug,
    )


def process_dir(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input-dir and output-dir must be different.")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(input_dir.glob(args.input_glob))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found: {input_dir}/{args.input_glob}")
    if args.max_files > 0:
        parquets = parquets[:args.max_files]

    jobs = []
    skipped = 0
    for input_path in parquets:
        output_path = output_dir / input_path.name
        if args.resume and output_path.exists():
            skipped += 1
            continue
        jobs.append((input_path, output_path))

    print(
        f"input={input_dir} output={output_dir} files={len(parquets)} "
        f"jobs={len(jobs)} skipped={skipped} file_workers={args.file_workers} "
        f"batch_rows={args.batch_rows} read_queue={args.read_queue_batches} "
        f"write_queue={args.write_queue_batches} compression={args.compression}",
        flush=True,
    )
    if not jobs:
        print("No pending files.", flush=True)
        return

    total_rows = 0
    completed = 0
    if args.file_workers <= 1:
        iterator = tqdm(jobs, total=len(jobs), desc="parquet files")
        for input_path, output_path in iterator:
            result = _process_one_serial(input_path, output_path, args)
            completed += 1
            total_rows += int(result["rows_written"])
            print(
                f"[done] {result['output']} rows={result['rows_written']} "
                f"batches={result['batches_written']} seconds={result['seconds']:.1f}",
                flush=True,
            )
            if result.get("debug"):
                print(f"[debug] {result['output']}: {result['debug']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.file_workers))) as executor:
            future_map = {
                _submit_file(executor, input_path, output_path, args): (input_path, output_path)
                for input_path, output_path in jobs
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="parquet files"):
                input_path, _ = future_map[future]
                result = future.result()
                completed += 1
                total_rows += int(result["rows_written"])
                print(
                    f"[done] {result['output']} rows={result['rows_written']} "
                    f"batches={result['batches_written']} seconds={result['seconds']:.1f}",
                    flush=True,
                )
                if result.get("debug"):
                    print(f"[debug] {result['output']}: {result['debug']}", flush=True)
                if int(result["rows_read"]) != int(result["rows_written"]):
                    raise RuntimeError(
                        f"row count mismatch for {input_path.name}: "
                        f"{result['rows_read']} != {result['rows_written']}"
                    )

    print(
        f"处理完成：done={completed} skipped={skipped} rows={total_rows} output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip self_audio from parquet training data with bounded streaming I/O."
    )
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--input-glob", default="*.parquet")
    parser.add_argument("--resume", action="store_true", help="skip existing final output files")
    parser.add_argument("--debug", action="store_true", help="print one before/after sample per file")
    parser.add_argument("--max-files", type=int, default=0, help="process only first N files")
    parser.add_argument("--file-workers", type=int, default=DEFAULT_FILE_WORKERS)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--read-queue-batches", type=int, default=DEFAULT_READ_QUEUE_BATCHES)
    parser.add_argument("--write-queue-batches", type=int, default=DEFAULT_WRITE_QUEUE_BATCHES)
    parser.add_argument("--compression", default=DEFAULT_COMPRESSION)
    args = parser.parse_args()
    process_dir(args)


if __name__ == "__main__":
    main()
