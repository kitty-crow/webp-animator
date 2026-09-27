from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import job_control


_installed = False


def _terminate(process: subprocess.Popen) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception:
        pass


def install(generative_vfi) -> None:
    """Make ResShift/MoG/ToonCrafter children obey the same job lifecycle as other engines.

    The original generative sharder launches Popen directly, and ThreadPoolExecutor
    worker threads do not inherit job_control's thread-local job id. As a result the
    Stop button could leave diffusion workers running in the background. Capture the
    parent render job id explicitly, register every child against it, and terminate
    sibling shards if one worker fails.
    """
    global _installed
    if _installed:
        return
    _installed = True

    def run_generative_interpolator(engine, tasks, stage_dir: Path, progress=None):
        if not tasks:
            return {}

        worker_dir = Path(stage_dir) / f"{engine}-interpolator-v2"
        worker_dir.mkdir(parents=True, exist_ok=True)

        gpu_tokens = generative_vfi._visible_gpu_tokens()
        requested = int(os.environ.get("GENERATIVE_VFI_GPU_WORKERS", "0") or 0)
        if requested > 0:
            gpu_tokens = gpu_tokens[:requested]
        if not gpu_tokens:
            gpu_tokens = [None]
        worker_count = max(1, min(len(tasks), len(gpu_tokens)))

        shards: list[list[dict]] = [[] for _ in range(worker_count)]
        for index, task in enumerate(tasks):
            shards[index % worker_count].append(task)

        job_id = job_control.current_job_id()
        progress_lock = threading.Lock()
        process_lock = threading.Lock()
        active_processes: set[subprocess.Popen] = set()
        state = {index: (0, 1) for index in range(worker_count)}

        def update(index: int, current: int, total: int, detail: str | None):
            with progress_lock:
                state[index] = (max(0, current), max(1, total))
                fraction = sum(min(1.0, c / t) for c, t in state.values()) / worker_count
                completed = sum(c for c, _ in state.values())
                aggregate_total = sum(t for _, t in state.values())
            if progress:
                suffix = f" · {worker_count} GPU worker{'s' if worker_count != 1 else ''}" if worker_count > 1 else ""
                label = generative_vfi.LABELS.get(engine, engine)
                message = f"{label} · {detail}{suffix}" if detail else f"{label} diffusion {completed}/{aggregate_total}{suffix}"
                progress(fraction, message)

        def run_shard(shard_index: int, shard_tasks: list[dict], gpu_token: str | None):
            shard_dir = worker_dir / f"shard-{shard_index:02d}"
            output_dir = shard_dir / "frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = shard_dir / "manifest.json"
            result = shard_dir / "result.json"
            manifest.write_text(
                json.dumps({"output_dir": str(output_dir), "tasks": shard_tasks}, indent=2),
                encoding="utf-8",
            )
            command, cwd = generative_vfi._engine_command(engine, manifest, result)
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            if gpu_token is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_token)

            if job_id:
                job_control.raise_if_cancelled(job_id)
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=str(cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with process_lock:
                active_processes.add(process)
            job_control.register_process(process, job_id)

            tail: list[str] = []
            current = 0
            total = 1
            try:
                assert process.stdout is not None
                for raw in process.stdout:
                    if job_id:
                        job_control.raise_if_cancelled(job_id)
                    line = raw.strip()
                    if line.startswith("PROGRESS "):
                        try:
                            _, current_text, total_text = line.split()
                            current = max(0, int(current_text))
                            total = max(1, int(total_text))
                            update(shard_index, current, total, None)
                        except job_control.JobCancelled:
                            raise
                        except Exception:
                            pass
                    elif line:
                        tail.append(line)
                        tail = tail[-40:]
                        if any(token in line.lower() for token in ("worker", "low-vram", "retry", "oom", "device=")):
                            update(shard_index, current, total, line)

                code = process.wait()
                if job_id:
                    job_control.raise_if_cancelled(job_id)
                if code != 0:
                    details = "\n".join(tail[-14:]) or "No worker diagnostics."
                    label = generative_vfi.LABELS.get(engine, engine)
                    raise RuntimeError(f"{label} worker failed:\n{details}")
                if not result.is_file():
                    raise RuntimeError(f"{generative_vfi.LABELS.get(engine, engine)} worker exited without a result manifest.")
                value = json.loads(result.read_text(encoding="utf-8"))
                return list(value.get("tasks", []))
            finally:
                job_control.unregister_process(process, job_id)
                with process_lock:
                    active_processes.discard(process)
                if process.poll() is None:
                    _terminate(process)

        outputs: list[dict] = []
        try:
            if worker_count == 1:
                outputs.extend(run_shard(0, shards[0], gpu_tokens[0]))
            else:
                pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{engine}-gpu")
                futures = {
                    pool.submit(run_shard, index, shard, gpu_tokens[index]): index
                    for index, shard in enumerate(shards)
                    if shard
                }
                try:
                    for future in as_completed(futures):
                        outputs.extend(future.result())
                except Exception:
                    with process_lock:
                        running = list(active_processes)
                    for process in running:
                        _terminate(process)
                    for future in futures:
                        future.cancel()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        finally:
            with process_lock:
                running = list(active_processes)
            for process in running:
                _terminate(process)

        if progress:
            progress(1.0, f"{generative_vfi.LABELS.get(engine, engine)} interpolation complete")
        return {str(item.get("id")): dict(item) for item in outputs}

    run_generative_interpolator.__generative_process_control__ = True
    generative_vfi.run_generative_interpolator = run_generative_interpolator
