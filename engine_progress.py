from __future__ import annotations

import threading


_state = threading.local()
_installed = False


def install(advanced_pipeline_module) -> None:
    """Preserve worker x/y progress in the persistent job status.

    Generator workers already print ``PROGRESS current total``. The temporal
    pipeline converts those values into a fractional progress callback, which is
    useful for the main progress bar but used to discard the concrete frame count.
    This adapter keeps the existing percentage mapping intact while making the
    synchronous legacy.set_job call inside that callback aware of the worker's
    current/total counts.
    """
    global _installed
    if _installed:
        return
    _installed = True

    import app as legacy

    original_run_process = advanced_pipeline_module._run_process
    original_set_job = legacy.set_job

    def run_process(command, *, cwd, progress_callback=None, label: str):
        if progress_callback is None or str(label).upper() not in {"SPEED", "EDEN"}:
            return original_run_process(
                command,
                cwd=cwd,
                progress_callback=progress_callback,
                label=label,
            )

        def counted_progress(current, total):
            _state.engine = str(label).upper()
            _state.current = max(0, int(current))
            _state.total = max(1, int(total))
            try:
                return progress_callback(current, total)
            finally:
                _state.engine = None
                _state.current = None
                _state.total = None

        return original_run_process(
            command,
            cwd=cwd,
            progress_callback=counted_progress,
            label=label,
        )

    def set_job(job_id: str, **changes):
        engine = getattr(_state, "engine", None)
        current = getattr(_state, "current", None)
        total = getattr(_state, "total", None)
        message = changes.get("message")
        if (
            engine in {"SPEED", "EDEN"}
            and current is not None
            and total is not None
            and isinstance(message, str)
            and engine in message.upper()
            and "FRAME " not in message.upper()
        ):
            changes = dict(changes)
            changes["message"] = f"{message} · frame {current}/{total}"
        return original_set_job(job_id, **changes)

    advanced_pipeline_module._run_process = run_process
    legacy.set_job = set_job
